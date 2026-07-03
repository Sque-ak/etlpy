import httpx
from etl.generic import Step, Data


class AuthenticateBasic(Step):
    """
    HTTP Basic (RFC 7617): token is empty, header is static.
    """

    def __init__(self, user: str, password: str, headers: dict | None = None, timeout: float = 60.0):
        self.user, self.password, self.timeout = user, password, timeout
        self.headers = headers or {}

    async def apply(self, df, data: Data = None):
        client = data.get("client") or httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        client.auth  = httpx.BasicAuth(self.user, self.password) # base64 (user:password)
        client.headers.update(self.headers)

        data["client"] = client
        return df