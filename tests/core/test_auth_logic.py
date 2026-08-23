"""Unit tests for the pure auth logic in core.auth (no FastAPI, no settings)."""

import json

from core.auth import Principal, load_api_keys, resolve_principal


def test_load_api_keys_empty_inputs():
    assert load_api_keys() == {}
    assert load_api_keys(None, None) == {}
    assert load_api_keys(None, "   ") == {}


def test_load_api_keys_json_shorthand_and_full_shapes():
    store = load_api_keys(
        None,
        json.dumps(
            {
                "key-alice": "alice",  # shorthand -> user_id == label
                "key-bob": {"user_id": "bob", "label": "Bob"},
                "key-bare": {"user_id": "carol"},  # label falls back to user_id
            }
        ),
    )
    assert set(store) == {"key-alice", "key-bob", "key-bare"}
    alice = store["key-alice"]
    assert isinstance(alice, Principal)
    assert alice.user_id == "alice"
    assert alice.label == "alice"
    assert alice.source == "api_key"
    assert alice.is_identified is True

    bob = store["key-bob"]
    assert bob.user_id == "bob"
    assert bob.label == "Bob"

    carol = store["key-bare"]
    assert carol.user_id == "carol"
    assert carol.label == "carol"  # falls back to user_id


def test_load_api_keys_skips_malformed_entries():
    store = load_api_keys(
        None,
        json.dumps(
            {
                "good": "good-user",
                "": "empty-key",  # empty key skipped
                "no-uid": {"label": "no uid"},  # missing user_id skipped
                "wrong-type": 42,  # non-str/non-dict value skipped
            }
        ),
    )
    assert list(store) == ["good"]


def test_load_api_keys_malformed_json_returns_empty():
    assert load_api_keys(None, "not json {") == {}
    assert load_api_keys(None, "[1, 2, 3]") == {}  # not an object


def test_load_api_keys_from_file(tmp_path):
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"file-key": {"user_id": "dave", "label": "Dave"}}), encoding="utf-8")
    store = load_api_keys(str(path))
    assert store["file-key"].user_id == "dave"


def test_load_api_keys_missing_file_returns_empty():
    assert load_api_keys("/no/such/path/keys.json") == {}


def test_resolve_principal_nothing_configured_is_anonymous():
    principal = resolve_principal(token=None, api_keys={}, auth_secret=None)
    assert principal is not None
    assert principal.source == "anonymous"
    assert principal.is_identified is False
    # A presented but unexpected token is still anonymous when nothing is configured
    principal2 = resolve_principal(token="whatever", api_keys={}, auth_secret=None)
    assert principal2.source == "anonymous"


def test_resolve_principal_auth_secret_only():
    principal = resolve_principal(token="s3cr3t", api_keys={}, auth_secret="s3cr3t")
    assert principal.source == "auth_secret"
    assert principal.user_id == "default"
    assert principal.is_identified is False  # shared secret does not identify a user
    # Wrong token with auth configured -> unauthorized
    assert resolve_principal(token="nope", api_keys={}, auth_secret="s3cr3t") is None
    # Missing token with auth configured -> unauthorized
    assert resolve_principal(token=None, api_keys={}, auth_secret="s3cr3t") is None


def test_resolve_principal_api_key_takes_precedence_over_secret():
    api_keys = {"alice-key": Principal("alice", "Alice", "api_key")}
    # A key that also equals the secret still resolves to the api-key principal
    principal = resolve_principal(token="alice-key", api_keys=api_keys, auth_secret="alice-key")
    assert principal.source == "api_key"
    assert principal.user_id == "alice"
    # The shared secret still works for the shared default principal
    principal2 = resolve_principal(token="shared", api_keys=api_keys, auth_secret="shared")
    assert principal2.source == "auth_secret"
    assert principal2.user_id == "default"
    # Unknown token rejected when keys+secret configured
    assert resolve_principal(token="bogus", api_keys=api_keys, auth_secret="shared") is None


def test_resolve_principal_api_keys_only_no_secret():
    api_keys = {"bob-key": Principal("bob", "Bob", "api_key")}
    assert resolve_principal(token="bob-key", api_keys=api_keys, auth_secret=None).user_id == "bob"
    assert resolve_principal(token="bob-key", api_keys=api_keys, auth_secret="") is not None
    assert resolve_principal(token=None, api_keys=api_keys, auth_secret=None) is None
