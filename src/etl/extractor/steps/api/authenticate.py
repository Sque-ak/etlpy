from __future__ import annotations
from dataclasses import dataclass, asdict
import httpx

from etl.generic.step import Step
from etl.generic.context import Data

@dataclass
class AuthFields:
    """
    Map fields with the Authenticate step can works.

    Where each OAuth 2.0 token-response field lives (RFC 6749 section 5.1).
    Value = dotted path in th response; None = The API does not return it.

    AuthFields(
        access_token="data.access_token",
        expires_in="data.expires_in",
        refresh_token="data.refresh_token",
    )
    """
    access_token: str = "data.access_token"
    token_type: str | None = None
    expires_in: str | None = "data.expires_in"
    refresh_token: str | None = "data.refresh_token"
    scope: str | None = None


class Authenticate(Step):
    """
    Generic API auth: send credentials, extract  a token, configure the shared
    client and stash the auth in data.

    Implements the non-interactive OAuth 2.0 token-endpoint flows, primarily the
    Client Credentials grant (RFC 6749 section 4.4). The access token is applied
    per RFC 6750 (Bearer). Client authentication via request body or HTTP Basic
    (RFC 7617) is selectable through 'send'.

    - RFC 6749  OAuth 2.0 Authorization Framework  https://www.rfc-editor.org/info/rfc6749/
    - RFC 6750  OAuth 2.0 Bearer Token Usage       https://www.rfc-editor.org/info/rfc6750/
    - RFC 7617  HTTP Basic Authentication          https://www.rfc-editor.org/info/rfc7617/
    - RFC 9700  OAuth 2.0 Security BCP             https://www.rfc-editor.org/info/rfc9700/
    """

    def __init__(
            self,
            url: str,
            credentials: dict,
            fields: AuthFields = None, 
            send="json",
            method="POST",
            auth_header=None,
            store="auth"
    ):
        self.url = url
        self.credentials = credentials
        self.fields = fields or AuthFields()
        self.send = send
        self.method = method
        self.auth_header = auth_header
        self.store = store

    async def apply(self, df, data: Data):
        client = data.get("client") or httpx.AsyncClient()
        response = await client.request(self.method, self.url, **{self.send:self.credentials})
        response.raise_for_status()

        body = response.json()
        auth = {}
        for name, path in asdict(self.fields).items():
            if path is None: # API does not return this field
                print(f"API does not return this field: {name}")
                continue
            auth[name] = self._get(body, path, required=(name == "access_token"))

        token = auth["access_token"]
        headers = {key: value.format(token=token) for key, value in self.auth_header.items()}
        client.headers.update(headers) # downstream uses data["client"]

        data["client"] = client
        data[self.store] = auth
        return df
    
    def _get(self, body, path, required=True):
        obj = body
        for key in path.split("."):
            if not isinstance(obj, dict) or key not in obj:
                if required:
                    raise KeyError(f"path '{path}' not found in auth response")
                return None
            obj = obj[key]
        return obj 