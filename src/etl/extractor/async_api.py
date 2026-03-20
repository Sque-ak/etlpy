from abc import abstractmethod
import httpx, logging
from datetime import datetime

from etl.extractor.async_extractor import AsyncSource


class AsyncAPI(AsyncSource):
    """
    Abstract base class for async API objects.

    Async counterpart of :class:`API`. Uses ``httpx.AsyncClient`` for
    non-blocking HTTP requests.

    Example::

        class MyBankAPI(AsyncAPI):
            async def authenticate(self):
                resp = await self.client.post(f"{self.url}/auth", json={...})
                self.access_token = resp.json()["token"]
                self.header = {"Authorization": f"Bearer {self.access_token}"}

            async def extract(self):
                resp = await self.endpoint(
                    self.client.get(f"{self.url}/data", headers=self.header)
                )
                return resp.json()
    """

    name: str
    url: str
    access_token: str
    refresh_token: str
    client: httpx.AsyncClient
    logger: logging.Logger
    header: dict[str, str] = None

    try_auth_error: int = 0
    time_to_block: datetime = None

    CONNECT = 10.0
    READ = 600.0
    WRITE = 30.0
    POOL = 10.0

    def __init__(
        self,
        name=None,
        url=None,
        header=None,
        access_token=None,
        refresh_token=None,
        client=None,
        logger=None,
    ) -> None:
        """Initialize the async API object with the provided parameters."""
        super().__init__()

        self.name = name
        self.url = url
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.header = header

        if client:
            self.client = client
        else:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.CONNECT,
                    read=self.READ,
                    write=self.WRITE,
                    pool=self.POOL,
                ),
                verify=True,
            )

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(f"extractor.{self.name}")

        self.try_auth_error = 0
        self.time_to_block = None

    @abstractmethod
    async def authenticate(self) -> None:
        """Async method for authenticating with the API service."""
        pass

    async def refresh_auth_token(self) -> None:
        """Overridable async method for refreshing the authentication token."""
        pass

    async def endpoint(self, request, skip_auth: bool = False) -> httpx.Response:
        """
        Async method for making requests to the API service with authentication.

        Args:
            request: An awaitable httpx request (e.g. ``self.client.get(...)``).
            skip_auth: Flag to skip authentication. Defaults to False.

        Returns:
            httpx.Response or None on auth/HTTP errors.
        """
        if self.time_to_block and (datetime.now() - self.time_to_block).seconds >= 3600:
            self.try_auth_error = 0
            self.time_to_block = None
            self.logger.info("Reset block for %s after 1 hour.", self.name)

        if self.try_auth_error >= 3 and (
            not self.time_to_block
            or (datetime.now() - self.time_to_block).seconds < 3600
        ):
            self.logger.error(
                "Exceeded authentication attempts for %s, skipping for today", self.name
            )
            if not self.time_to_block:
                self.time_to_block = datetime.now()
            return None

        if not self.access_token and not skip_auth:
            await self.refresh_auth_token()
            await self.authenticate()
            return await self.endpoint(request, skip_auth)

        try:
            response = await request
            response.raise_for_status()
        except httpx.HTTPStatusError as http_err:
            if response.status_code == 401:
                self.try_auth_error += 1
                self.logger.warning(
                    "Authentication error in %s, attempt %d/3: %s",
                    self.name, self.try_auth_error, http_err,
                )
                return None
            else:
                self.logger.error("HTTP error in %s: %s", self.name, http_err)
                return None
        except httpx.RequestError as req_err:
            self.logger.error("Request error to %s: %s", self.name, req_err)
            raise
        except Exception as err:
            self.logger.error("Unknown error in %s: %s", self.name, err)
            raise

        return response

    async def close(self) -> None:
        """Close the underlying async HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
