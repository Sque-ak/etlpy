import httpx, pytest
import json

from etl.generic import Data, Pipeline
from etl.extractor.steps.api import Authenticate

def _auth_api(expected: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content) # what Authenticate actually sent (send="json")
        if sent == expected:
            return httpx.Response(200, json={"data": {"access_token":"abc123"}})
        return httpx.Response(401, json={"error": "invalid credentials"})
    return handler

def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))

async def test_authenticate_runs_in_pipeline():
    def handler(request):
        return httpx.Response(200, json={"data": {"access_token": "abc123"}})
    
    data = Data(client=_client(handler)) # Fake client
    pipe = Pipeline(
        [
            Authenticate(url="https://api.test/auth", 
                      credentials={}, 
                      auth_header={"Authorization": "Bearer {token}"})
        ],
        data=data,
    )

    await pipe.run()

    assert pipe.data["auth"]["access_token"] == "abc123"
    assert pipe.data["client"].headers["Authorization"] == "Bearer abc123"

async def test_correct_credentials_return_token():
    creds = {"login": "admin", "password": "secret"}
    data = Data(client=_client(_auth_api(creds)))
    
    pipe = Pipeline(
        [
            Authenticate(url="https://api.test/auth", 
                      credentials=creds, 
                      auth_header={"Authorization": "Bearer {token}"})
        ],
        data=data,
    )
    await pipe.run()

    assert data["auth"]["access_token"] == "abc123"
    assert data["client"].headers["Authorization"] == "Bearer abc123"

async def test_wrong_password_rejected():
    expected = {"login": "admin", "password": "secret"}
    data = Data(client=_client(_auth_api(expected)))

    pipe = Pipeline(
        [
            Authenticate(url="https://api.test/auth", 
                      credentials={"login": "admin", "password": "WRONG"}, 
                      auth_header={"Authorization": "Bearer {token}"})
        ],
        data=data,
    )

    with pytest.raises(httpx.HTTPStatusError):
        await pipe.run()
    assert "auth" not in data

async def test_api_key_accepted():
    data = Data(client=_client(_auth_api({"api_key": "KEY-123"})))

    pipe = Pipeline(
        [
            Authenticate(url="https://api.test/auth", 
                      credentials={"api_key": "KEY-123"},
                      auth_header={"Authorization": "Bearer {token}"})
        ],
        data=data,
    )

    await pipe.run()
    assert data["auth"]["access_token"] == "abc123"

def test_get_required_missing():
    step = Authenticate(url="http://x", credentials={})
    with pytest.raises(KeyError):
        step._get({"data": {}}, "data.access_token", required=True)