"""User identity resolution for the DocPilot API.

Backward-compatible with the single ``AUTH_SECRET`` bearer token: when no
per-user API keys are configured the shared secret (or anonymous access) is
used exactly as before. When ``AUTH_API_KEYS`` are configured, each key
identifies a distinct user whose ``user_id`` the server pins to the request
so clients cannot impersonate another user.

No new dependencies: JSON parsing and constant-time comparison come from the
standard library.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

PrincipalSource = Literal["api_key", "auth_secret", "anonymous"]


@dataclass(frozen=True)
class Principal:
    """The authenticated (or anonymous) caller of a single request."""

    user_id: str
    label: str
    source: PrincipalSource

    @property
    def is_identified(self) -> bool:
        """True when the principal carries a real per-user identity (api key).

        Only api-key principals are trusted to override the client-supplied
        ``user_id``; the shared ``AUTH_SECRET`` and anonymous modes fall back to
        whatever the client sent (preserving prior behavior).
        """
        return self.source == "api_key"


def load_api_keys(
    path: str | None = None, json_str: str | None = None
) -> dict[str, Principal]:
    """Parse the API-key store into ``{raw_key: Principal}``.

    ``json_str`` (e.g. ``AUTH_API_KEYS_JSON``) takes precedence over ``path``
    (``AUTH_API_KEYS_FILE``). Accepted JSON shapes:

        {"<key>": {"user_id": "alice", "label": "Alice"}}
        {"<key>": "alice"}            # shorthand: key -> user_id

    Malformed entries are skipped with a warning rather than aborting startup,
    so one bad key never locks out the whole API.
    """
    raw: str | None = None
    source_desc = ""
    if json_str:
        raw = json_str
        source_desc = "AUTH_API_KEYS_JSON"
    elif path:
        if not os.path.exists(path):
            logger.warning("AUTH_API_KEYS_FILE %s does not exist", path)
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                raw = fh.read()
            source_desc = path
        except OSError as e:  # pragma: no cover - filesystem error path
            logger.warning("Cannot read AUTH_API_KEYS_FILE %s: %s", path, e)
            return {}
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Invalid API-key JSON in %s: %s", source_desc, e)
        return {}
    if not isinstance(data, dict):
        logger.warning("API-key JSON in %s is not an object", source_desc)
        return {}

    store: dict[str, Principal] = {}
    for key, val in data.items():
        if not isinstance(key, str) or not key:
            continue
        if isinstance(val, str):
            uid = val
            label = val
        elif isinstance(val, dict):
            uid = str(val.get("user_id") or "")
            label = str(val.get("label") or uid)
        else:
            continue
        if not uid:
            continue
        store[key] = Principal(user_id=uid, label=label, source="api_key")
    return store


def resolve_principal(
    token: str | None,
    api_keys: dict[str, Principal],
    auth_secret: str | None,
) -> Principal | None:
    """Resolve the caller's principal from a presented bearer token.

    Returns ``None`` when auth is configured but the token is missing or
    invalid (the caller raises 401). When nothing is configured, returns an
    anonymous principal so the API stays open exactly as before.
    """
    configured = bool(api_keys) or bool(auth_secret)
    if not configured:
        return Principal(user_id="anonymous", label="anonymous", source="anonymous")
    if not token:
        return None
    # Per-user keys take precedence; compare the shared secret in constant time.
    principal = api_keys.get(token)
    if principal is not None:
        return principal
    if auth_secret and secrets.compare_digest(token, auth_secret):
        return Principal(user_id="default", label="default", source="auth_secret")
    return None
