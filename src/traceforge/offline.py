"""Runtime network prohibition for ``--offline``.

Replaces the socket primitives used for outbound IP connections so any attempt (including by a
dependency) raises :class:`NetworkBlockedError`.  AF_UNIX sockets are left alone.  This is a
best-effort in-process guard, not an OS sandbox.
"""

from __future__ import annotations

import socket

_ORIG: dict[str, object] = {}


class NetworkBlockedError(RuntimeError):
    pass


def _blocked(*_a, **_k):
    raise NetworkBlockedError("network access is disabled (--offline)")


def block_network() -> None:
    if _ORIG:
        return
    for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        _ORIG[name] = getattr(socket, name)
        setattr(socket, name, _blocked)
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    _ORIG["connect"] = orig_connect
    _ORIG["connect_ex"] = orig_connect_ex

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _blocked()
        return orig_connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _blocked()
        return orig_connect_ex(self, address)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]


def unblock_network() -> None:
    """Restore sockets (used by tests)."""
    for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        if name in _ORIG:
            setattr(socket, name, _ORIG[name])
    if "connect" in _ORIG:
        socket.socket.connect = _ORIG["connect"]  # type: ignore[method-assign]
        socket.socket.connect_ex = _ORIG["connect_ex"]  # type: ignore[method-assign]
    _ORIG.clear()
