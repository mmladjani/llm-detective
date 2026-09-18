"""Where a parked investigation lives between two HTTP requests.

The app used to hold live `GameSession` objects in a module-level dict. That works for
exactly one deployment shape: a single long-lived process. On a serverless platform
(Vercel) a function is not a persistent process — a second request may land on a
different instance, and instances are recycled. The dict then "works sometimes",
which is worse than failing, because the per-turn UI loses the investigation halfway
through with no error to point at.

So the session is serialized after every turn and reloaded before the next one, and
WHERE it is kept is a backend choice:

  * `MemoryStore` — a dict, exactly the old behaviour. Default for local runs and
    tests, where a single uvicorn process is the whole world.
  * `RedisStore` — Upstash Redis over its REST API, which is what Vercel's Marketplace
    Redis integration provisions (Vercel's own KV was retired into Upstash). REST,
    not a socket client, so it needs no native dependency and no connection pool to
    keep alive across invocations — `httpx` was already a dependency.

`build_store()` picks by environment, so nothing in the app has to know which one is
in play. Selection is logged once at import: a deployment silently falling back to
MemoryStore is the failure this module exists to prevent, and it must be visible.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional, Protocol

import httpx

log = logging.getLogger(__name__)

# How long a parked investigation survives. Long enough for a person to be interviewed
# and answer (the human-in-the-loop cases park on a real question and wait), short
# enough that abandoned sessions do not accumulate forever.
DEFAULT_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", str(2 * 60 * 60)))

# Upstash injects the first pair; Vercel's retired KV integration injected the second,
# and projects migrated from it still carry those names. Accept either rather than
# making the operator rename a variable the platform set for them.
_URL_VARS = ("UPSTASH_REDIS_REST_URL", "KV_REST_API_URL")
_TOKEN_VARS = ("UPSTASH_REDIS_REST_TOKEN", "KV_REST_API_TOKEN")


def _first_env(names: tuple[str, ...]) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


class SessionStore(Protocol):
    """Keys are opaque strings; values are JSON-serializable dicts."""

    name: str

    def get(self, key: str) -> Optional[dict[str, Any]]: ...

    def put(self, key: str, value: dict[str, Any],
            ttl: int = DEFAULT_TTL_SECONDS) -> None: ...

    def delete(self, key: str) -> None: ...

    def add_to_index(self, index_key: str, member: str,
                     ttl: int = DEFAULT_TTL_SECONDS) -> None: ...

    def members(self, index_key: str) -> list[str]: ...

    def incr(self, key: str, ttl: int) -> int: ...


class MemoryStore:
    """Process-local. Correct only where one process serves every request."""

    name = "memory"

    def __init__(self) -> None:
        # key -> (expires_at_epoch, value)
        self._data: dict[str, tuple[float, dict[str, Any]]] = {}
        self._indexes: dict[str, set[str]] = {}
        self._counters: dict[str, tuple[float, int]] = {}

    def _sweep(self) -> None:
        now = time.time()
        for key in [k for k, (exp, _) in self._data.items() if exp <= now]:
            del self._data[key]

    def get(self, key: str) -> Optional[dict[str, Any]]:
        self._sweep()
        hit = self._data.get(key)
        return None if hit is None else hit[1]

    def put(self, key: str, value: dict[str, Any],
            ttl: int = DEFAULT_TTL_SECONDS) -> None:
        self._data[key] = (time.time() + ttl, value)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def add_to_index(self, index_key: str, member: str,
                     ttl: int = DEFAULT_TTL_SECONDS) -> None:
        self._indexes.setdefault(index_key, set()).add(member)

    def members(self, index_key: str) -> list[str]:
        return sorted(self._indexes.get(index_key, set()))

    def incr(self, key: str, ttl: int) -> int:
        now = time.time()
        expires, count = self._counters.get(key, (0.0, 0))
        if expires <= now:
            expires, count = now + ttl, 0
        count += 1
        self._counters[key] = (expires, count)
        return count


class RedisStore:
    """Upstash Redis over REST.

    Commands go as a JSON array in the POST body (`["SET", key, value, "EX", ttl]`)
    rather than in the URL path, because a serialized session is far too large to put
    in a path segment.

    Read failures return None and write failures raise. That asymmetry is deliberate:
    a lost read looks like an expired session, which the app already handles with a
    404, while a silently dropped write would strand the caller mid-investigation with
    a session id that will never resolve.
    """

    name = "redis"

    def __init__(self, url: str, token: str, timeout: float = 5.0) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def _command(self, *args: Any) -> Any:
        payload = [str(a) for a in args]
        response = httpx.post(
            self._url,
            json=payload,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        body = response.json()
        if isinstance(body, dict) and body.get("error"):
            raise RuntimeError(f"Redis error: {body['error']}")
        return body.get("result") if isinstance(body, dict) else None

    def get(self, key: str) -> Optional[dict[str, Any]]:
        try:
            raw = self._command("GET", key)
        except Exception as exc:  # treated as a miss -> the caller 404s
            log.warning("session store read failed for %s: %s", key, exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError) as exc:
            # A corrupt value is a miss, not a crash; the alternative is a 500 the
            # user can do nothing about.
            log.warning("session store value for %s is not valid JSON: %s", key, exc)
            return None

    def put(self, key: str, value: dict[str, Any],
            ttl: int = DEFAULT_TTL_SECONDS) -> None:
        self._command("SET", key, json.dumps(value), "EX", ttl)

    def delete(self, key: str) -> None:
        try:
            self._command("DEL", key)
        except Exception as exc:  # the TTL will collect it anyway
            log.warning("session store delete failed for %s: %s", key, exc)

    def add_to_index(self, index_key: str, member: str,
                     ttl: int = DEFAULT_TTL_SECONDS) -> None:
        self._command("SADD", index_key, member)
        # Refreshed on every add. Individual members cannot carry their own TTL, so
        # readers must tolerate an id whose value has already expired — see
        # app._custom_case_ids, which drops ids that no longer resolve.
        self._command("EXPIRE", index_key, ttl)

    def members(self, index_key: str) -> list[str]:
        try:
            result = self._command("SMEMBERS", index_key)
        except Exception as exc:
            log.warning("session index read failed for %s: %s", index_key, exc)
            return []
        return sorted(str(m) for m in (result or []))

    def incr(self, key: str, ttl: int) -> int:
        """Atomic counter, used for the spend limit.

        INCR server-side rather than read-modify-write, so two concurrent function
        instances cannot both see the same count and both let a request through —
        which is the whole reason the limit lives in the store and not in memory.

        Failures raise. A rate limiter that fails open is not a rate limiter: if the
        store is unreachable the safe answer is to refuse the expensive operation.
        """
        count = int(self._command("INCR", key))
        if count == 1:
            # First hit in this window sets the window's lifetime. EXPIRE only on the
            # first increment, or every request would slide the window forward and the
            # counter would never reset under sustained traffic.
            self._command("EXPIRE", key, ttl)
        return count


def redis_configured() -> bool:
    return bool(_first_env(_URL_VARS) and _first_env(_TOKEN_VARS))


def build_store() -> SessionStore:
    url, token = _first_env(_URL_VARS), _first_env(_TOKEN_VARS)
    if url and token:
        log.info("session store: redis (%s)", url.split("//")[-1].split(".")[0])
        return RedisStore(url, token)
    if os.environ.get("VERCEL"):
        # Vercel sets VERCEL=1 in every deployment. Reaching here means the app is
        # serverless with a process-local store, so the per-turn UI will drop
        # investigations non-deterministically. Say so loudly at boot rather than
        # letting it be discovered as a flaky bug.
        log.error(
            "session store: MEMORY on a serverless deployment. Sessions will NOT "
            "survive between requests. Attach a Redis integration and set "
            "UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN."
        )
    else:
        log.info("session store: memory (single process)")
    return MemoryStore()
