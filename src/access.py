"""Access gate and spend limit for a publicly reachable deployment.

The app has no user model and does not want one: it is a teaching demo. But a public
URL plus a server-side `ANTHROPIC_API_KEY` is a stranger's licence to spend your
credits, and Vercel's free Deployment Protection only admits Vercel accounts that
already have access to the project — no good for handing a link to a colleague.

So there are two independent controls here, because they stop different things:

  * `require_access` answers "may you talk to this app at all". HTTP Basic, so the
    browser prompts once and then attaches the header to every `fetch` the UI makes;
    no login page, no cookie handling, no session of our own to secure.
  * `check_llm_budget` answers "how much may the world spend today". A single
    investigation is already bounded (the case budget caps world actions and
    MAX_MODEL_CALLS caps model calls per turn), so the unbounded axis is the NUMBER of
    investigations. That is what gets counted.

Both are off when unconfigured, so local development and the offline test suite behave
exactly as before.
"""

from __future__ import annotations

import base64
import binascii
import os
import secrets
from typing import Optional

# Paths that stay reachable without credentials. Only the health check: it returns
# three booleans and a backend name, which is what makes a misconfigured deployment
# diagnosable with one curl, and reveals nothing worth protecting.
PUBLIC_PATHS = frozenset({"/api/config"})

DEFAULT_USER = "detective"


def access_password() -> Optional[str]:
    value = os.environ.get("APP_ACCESS_PASSWORD", "").strip()
    return value or None


def access_user() -> str:
    return os.environ.get("APP_ACCESS_USER", "").strip() or DEFAULT_USER


def gate_enabled() -> bool:
    return access_password() is not None


def llm_session_limit() -> int:
    """Max LLM investigations per window. 0 disables the limit."""
    try:
        return max(0, int(os.environ.get("MAX_LLM_SESSIONS_PER_HOUR", "0")))
    except ValueError:
        return 0


def llm_window_seconds() -> int:
    try:
        return max(60, int(os.environ.get("LLM_LIMIT_WINDOW_SECONDS", "3600")))
    except ValueError:
        return 3600


def _decode_basic(header: str) -> Optional[tuple[str, str]]:
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    user, sep, password = raw.partition(":")
    if not sep:
        return None
    return user, password


def credentials_ok(authorization: Optional[str]) -> bool:
    """Whether an Authorization header carries the configured credentials.

    Both fields are compared with compare_digest. The username is not a secret, but
    comparing it in constant time too keeps the check free of an early return that
    would leak which half was wrong.
    """
    password = access_password()
    if password is None:
        return True
    if not authorization:
        return False
    decoded = _decode_basic(authorization)
    if decoded is None:
        return False
    user, supplied = decoded
    user_ok = secrets.compare_digest(user, access_user())
    password_ok = secrets.compare_digest(supplied, password)
    return user_ok and password_ok


def is_public(path: str) -> bool:
    return path in PUBLIC_PATHS


def llm_budget_key(window: int) -> str:
    """One counter per window, so a window expires instead of being reset by hand."""
    import time
    return f"trace:llm_sessions:{int(time.time()) // window}"
