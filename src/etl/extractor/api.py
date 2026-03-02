from abc import abstractmethod
import httpx, logging
from datetime import datetime

from etl.extractor import Source

class API(Source):
    """
    Abstract base class for API objects.

    This class provides a template for API objects, including authentication,
    token refresh, and request handling with error management.
    """

    name: str
    url: str
    access_token: str
    refresh_token: str
    client: httpx.Client
    logger: logging.Logger
    header: dict[str, str] = None

    try_auth_error: int = 0
    time_to_block: datetime = None

    CONNECT = 10.0
    READ = 600.0
    WRITE = 30.0
    POOL = 10.0

    def __init__(self, name = None, url = None, header = None, access_token = None, refresh_token = None, client = None, logger = None) -> None:
        """Initialize the API object with the provided parameters."""
        
        super().__init__()

        self.name = name
        self.url = url
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.header = header

        if client:
            self.client = client
        else:
            self.client = httpx.Client(    
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
    def authenticate(self) -> None:
        """Method for authenticating with the API service."""
        pass

    def refresh_token(self) -> None:
        """Overridable method for refreshing the authentication token."""
        pass

    def endpoint(self, request:any, skip_auth: bool = False) -> httpx.Response:
        """
        Method for making requests to the API service with authentication.

        Args:
            request (any): The request object to be sent to the API service.
            skip_auth (bool, optional): Flag to indicate whether to skip authentication. Defaults to False.
        
        Returns:
            httpx.Response: The response from the API service. 
        """
        
        if self.time_to_block and (datetime.now() - self.time_to_block).seconds >= 3600:
            self.try_auth_error = 0
            self.time_to_block = None
            self.logger.info(f"Reset block for {self.name} after 1 hour.")

        if self.try_auth_error >= 3 and (not self.time_to_block or (datetime.now() - self.time_to_block).seconds < 3600):
            self.logger.error(f"Exceeded authentication attempts for {self.name}, skipping for today")
            
            if not self.time_to_block:
                self.time_to_block = datetime.now()
            return
       
        if not self.access_token and not skip_auth:
            self.refresh_token()
            self.authenticate()
            return self.endpoint(request, skip_auth)

        try:
            response = request
            response.raise_for_status()
        except httpx.HTTPStatusError as http_err:
            if response.status_code == 401:
                self.try_auth_error += 1
                self.logger.warning(f"Authentication error in {self.name}, attempt {self.try_auth_error}/3: {http_err}")
                return
            else:
                self.logger.error(f"HTTP error in {self.name}: {http_err}")
                return
        except httpx.RequestError as req_err:
            self.logger.error(f"Request error to {self.name}: {req_err}")
            raise
        except Exception as err:
            self.logger.error(f"Unknown error in {self.name}: {err}")
            raise
        
        return response