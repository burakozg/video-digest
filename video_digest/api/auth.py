"""Admin API authentication (§9).

Ported verbatim from podcast_agent/api/auth.py (podcast-digest@fac8a1f),
except the env var name in the 503 message. See plan §1.5/§1.7.

Single shared key in the ``X-API-Key`` header, compared in constant time. The key
is read from ``app.state`` so tests can build an app without touching the process
environment.

Repeated failures from one address are then throttled. Constant-time comparison
defeats timing attacks and says nothing about volume, and this service listens on
the LAN with no other rate limit in front of it — one compromised device is
enough to sit and guess. The counter is deliberately in memory: an
unauthenticated request must not be able to make the service write to its
database, which is a denial-of-service in its own right.
"""

from __future__ import annotations

import hmac
import time
from collections import OrderedDict

from fastapi import Header, HTTPException, Request, status

API_KEY_HEADER = "X-API-Key"

#: Failures from one address before it is made to wait.
MAX_FAILURES = 10

#: How long failures are remembered, and how long a throttled address waits.
WINDOW_S = 300

#: Addresses tracked at once. A bounded dict, because the key of this map is
#: chosen by whoever is connecting: unbounded, a spray of forged sources would
#: grow it without limit. Oldest is evicted first.
MAX_TRACKED = 1024

#: address -> the times it failed, most recent last.
_failures: OrderedDict[str, list[float]] = OrderedDict()


def _recent(address: str, now: float) -> list[float]:
    """Failures still inside the window, pruned in place."""
    times = [t for t in _failures.get(address, []) if now - t < WINDOW_S]
    if times:
        _failures[address] = times
        _failures.move_to_end(address)
    else:
        _failures.pop(address, None)
    return times


def _record_failure(address: str, now: float) -> None:
    times = _recent(address, now)
    times.append(now)
    _failures[address] = times
    _failures.move_to_end(address)
    while len(_failures) > MAX_TRACKED:
        _failures.popitem(last=False)


def reset_throttle() -> None:
    """Forget every recorded failure. For tests, and for a fresh process."""
    _failures.clear()


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """Reject any request without a matching admin key."""
    expected: str | None = getattr(request.app.state, "admin_api_key", None)
    if not expected:
        # Fail closed: an unset key must never mean "open to everyone".
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="admin API key is not configured; set VIDEODIGEST_ADMIN_API_KEY",
        )

    address = request.client.host if request.client else "unknown"
    now = time.monotonic()
    if len(_recent(address, now)) >= MAX_FAILURES:
        # Deliberately says nothing about the key that was offered. The wait is
        # the same whether it was close, empty, or absent.
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="too many failed attempts; wait and try again",
            headers={"Retry-After": str(WINDOW_S)},
        )

    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        _record_failure(address, now)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
    # A correct key clears the address: a person who mistyped it twice and then
    # got it right is not mid-attack, and should not meet a wait later.
    _failures.pop(address, None)
