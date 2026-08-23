import json

import pytest
from pydantic import SecretStr

from core.ratelimit import rate_limiter


def _reset_auth_state():
    """Clear the api-key cache and rate-limit buckets between tests."""
    from service import service

    service._API_KEY_CACHE = None
    rate_limiter.reset()


@pytest.fixture(autouse=True)
def _isolate_auth_state():
    _reset_auth_state()
    yield
    _reset_auth_state()


def test_no_auth_secret(mock_settings, mock_agent, test_client):
    """Test that when AUTH_SECRET is not set, all requests are allowed"""
    mock_settings.AUTH_SECRET = None
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer any-token"},
    )
    assert response.status_code == 200

    # Should also work without any auth header
    response = test_client.post("/invoke", json={"message": "test"})
    assert response.status_code == 200


def test_auth_secret_correct(mock_settings, mock_agent, test_client):
    """Test that when AUTH_SECRET is set, requests with correct token are allowed"""
    mock_settings.AUTH_SECRET = SecretStr("test-secret")
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer test-secret"},
    )
    assert response.status_code == 200


def test_auth_secret_incorrect(mock_settings, mock_agent, test_client):
    """Test that when AUTH_SECRET is set, requests with wrong token are rejected"""
    mock_settings.AUTH_SECRET = SecretStr("test-secret")
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer wrong-secret"},
    )
    assert response.status_code == 401

    # Should also reject requests with no auth header
    response = test_client.post("/invoke", json={"message": "test"})
    assert response.status_code == 401


def test_api_key_pins_user_id_overrides_client(mock_settings, mock_agent, test_client):
    """An api-key caller's user_id is pinned server-side; a spoofed client id is ignored."""
    mock_settings.AUTH_SECRET = None
    mock_settings.AUTH_API_KEYS_JSON = json.dumps({"alice-key": {"user_id": "alice", "label": "Alice"}})
    response = test_client.post(
        "/invoke",
        # Client tries to impersonate a different user.
        json={"message": "hi", "user_id": "impostor"},
        headers={"Authorization": "Bearer alice-key"},
    )
    assert response.status_code == 200
    # The agent received the server-pinned user_id, not the client's "impostor".
    config = mock_agent.ainvoke.call_args.kwargs["config"]
    assert config["configurable"]["user_id"] == "alice"


def test_auth_secret_keeps_client_user_id(mock_settings, mock_agent, test_client):
    """The shared AUTH_SECRET does not identify a user, so the client user_id is honored (back-compat)."""
    mock_settings.AUTH_SECRET = SecretStr("shared-secret")
    response = test_client.post(
        "/invoke",
        json={"message": "hi", "user_id": "client-bob"},
        headers={"Authorization": "Bearer shared-secret"},
    )
    assert response.status_code == 200
    config = mock_agent.ainvoke.call_args.kwargs["config"]
    assert config["configurable"]["user_id"] == "client-bob"


def test_api_keys_from_file(mock_settings, mock_agent, test_client, tmp_path):
    """API keys load from a gitignored JSON file path."""
    mock_settings.AUTH_SECRET = None
    path = tmp_path / "api_keys.json"
    path.write_text(json.dumps({"file-key": "dave"}), encoding="utf-8")
    mock_settings.AUTH_API_KEYS_FILE = str(path)
    response = test_client.post(
        "/invoke",
        json={"message": "hi"},
        headers={"Authorization": "Bearer file-key"},
    )
    assert response.status_code == 200
    config = mock_agent.ainvoke.call_args.kwargs["config"]
    assert config["configurable"]["user_id"] == "dave"
    # Wrong key is rejected.
    bad = test_client.post(
        "/invoke",
        json={"message": "hi"},
        headers={"Authorization": "Bearer nope"},
    )
    assert bad.status_code == 401


def test_rate_limit_returns_429(mock_settings, mock_agent, test_client):
    """RATE_LIMIT_PER_MIN caps requests per caller with a Retry-After header."""
    mock_settings.AUTH_SECRET = None
    mock_settings.RATE_LIMIT_PER_MIN = 2
    r1 = test_client.post("/invoke", json={"message": "a"})
    r2 = test_client.post("/invoke", json={"message": "b"})
    r3 = test_client.post("/invoke", json={"message": "c"})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers


def test_rate_limit_independent_per_user(mock_settings, mock_agent, test_client):
    """Two api-key users get separate buckets."""
    mock_settings.AUTH_API_KEYS_JSON = json.dumps(
        {"alice-key": "alice", "bob-key": "bob"}
    )
    mock_settings.RATE_LIMIT_PER_MIN = 1
    # Alice uses her one request; Bob's bucket is untouched.
    assert test_client.post(
        "/invoke", json={"message": "a"}, headers={"Authorization": "Bearer alice-key"}
    ).status_code == 200
    assert test_client.post(
        "/invoke", json={"message": "b"}, headers={"Authorization": "Bearer alice-key"}
    ).status_code == 429
    assert test_client.post(
        "/invoke", json={"message": "c"}, headers={"Authorization": "Bearer bob-key"}
    ).status_code == 200

