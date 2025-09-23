import pytest

pytest.importorskip("multipart")

from backend.app import app
from fastapi.testclient import TestClient


def test_lan_info_endpoint_returns_urls():
    client = TestClient(app)
    response = client.get("/api/network/lan-info")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert isinstance(data.get("adminUrls"), list)
    assert isinstance(data.get("displayUrls"), list)
    assert len(data["adminUrls"]) >= 1
    assert len(data["displayUrls"]) >= 1
    for url in data["adminUrls"] + data["displayUrls"]:
        assert isinstance(url, str)
        assert url.startswith("http")
