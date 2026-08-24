from __future__ import annotations

import hashlib
import hmac
import json
import socket
import struct
from typing import Any

PROTOCOL_VERSION = 1
_HEADER_PREFIX = struct.Struct("!I")
_MAX_HEADER_BYTES = 64 * 1024
_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("TCP peer closed the connection.")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_frame(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    (header_size,) = _HEADER_PREFIX.unpack(_recv_exact(sock, _HEADER_PREFIX.size))
    if header_size <= 0 or header_size > _MAX_HEADER_BYTES:
        raise ValueError(f"Invalid frame header size: {header_size}.")
    header = json.loads(_recv_exact(sock, header_size).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("Frame header must be a JSON object.")
    if int(header.get("version", -1)) != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported protocol version {header.get('version')!r}."
        )
    payload_size = int(header.get("payload_size", 0))
    if payload_size < 0 or payload_size > _MAX_PAYLOAD_BYTES:
        raise ValueError(f"Invalid frame payload size: {payload_size}.")
    return header, _recv_exact(sock, payload_size)


def send_frame(
    sock: socket.socket,
    header: dict[str, Any],
    payload: bytes = b"",
) -> None:
    wire_header = dict(header)
    wire_header["version"] = PROTOCOL_VERSION
    wire_header["payload_size"] = len(payload)
    encoded_header = json.dumps(
        wire_header,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    if len(encoded_header) > _MAX_HEADER_BYTES:
        raise ValueError("Frame header is too large.")
    sock.sendall(_HEADER_PREFIX.pack(len(encoded_header)))
    sock.sendall(encoded_header)
    if payload:
        sock.sendall(payload)


def heartbeat_signature(token: str, session_id: str, counter: int) -> str:
    message = f"{session_id}:{counter}".encode("utf-8")
    return hmac.new(token.encode("utf-8"), message, hashlib.sha256).hexdigest()
