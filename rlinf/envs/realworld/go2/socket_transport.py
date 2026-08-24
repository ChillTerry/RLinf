# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import socket
import struct
import threading
import time
from typing import Any

import numpy as np

from .go2_vln_env import (
    Go2VLNConfig,
    NavPrimitive,
    PrimitiveResult,
    PrimitiveStatus,
)

PROTOCOL_VERSION = 1
_HEADER_PREFIX = struct.Struct("!I")
_MAX_HEADER_BYTES = 64 * 1024
_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Go2 socket gateway closed the TCP connection.")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_frame(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    (header_size,) = _HEADER_PREFIX.unpack(_recv_exact(sock, _HEADER_PREFIX.size))
    if header_size <= 0 or header_size > _MAX_HEADER_BYTES:
        raise ValueError(f"Invalid Go2 socket header size: {header_size}.")
    header = json.loads(_recv_exact(sock, header_size).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("Go2 socket frame header must be a JSON object.")
    if int(header.get("version", -1)) != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported Go2 socket protocol version {header.get('version')!r}."
        )
    payload_size = int(header.get("payload_size", 0))
    if payload_size < 0 or payload_size > _MAX_PAYLOAD_BYTES:
        raise ValueError(f"Invalid Go2 socket payload size: {payload_size}.")
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
        raise ValueError("Go2 socket frame header is too large.")
    sock.sendall(_HEADER_PREFIX.pack(len(encoded_header)))
    sock.sendall(encoded_header)
    if payload:
        sock.sendall(payload)


def heartbeat_signature(token: str, session_id: str, counter: int) -> str:
    message = f"{session_id}:{counter}".encode("utf-8")
    return hmac.new(token.encode("utf-8"), message, hashlib.sha256).hexdigest()


class SocketGo2VLNTransport:
    """Reliable TCP action/image transport with an independent UDP watchdog."""

    def __init__(self, config: Go2VLNConfig):
        self._config = config
        self._token = os.environ.get(config.socket_auth_token_env, "")
        if config.socket_require_auth and not self._token:
            raise RuntimeError(
                f"{config.socket_auth_token_env} must be set when "
                "socket_require_auth=true."
            )

        self._session_id = secrets.token_hex(16)
        self._request_id = 0
        self._request_lock = threading.Lock()
        self._closed = False
        self._heartbeat_stop = threading.Event()

        self._socket = socket.create_connection(
            (config.socket_host, config.socket_tcp_port),
            timeout=config.socket_connect_timeout_sec,
        )
        self._configure_tcp_keepalive(self._socket)
        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        try:
            response, _ = self._request(
                {
                    "type": "hello",
                    "session_id": self._session_id,
                    "token": self._token,
                },
                timeout_sec=config.socket_connect_timeout_sec,
            )
            if response.get("type") != "hello_result" or not response.get(
                "ok", False
            ):
                raise PermissionError(
                    str(response.get("message", "Go2 gateway rejected handshake."))
                )
        except Exception:
            self._socket.close()
            self._udp_socket.close()
            raise

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="rlinf-go2-udp-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def wait_for_image(
        self, *, timeout_sec: float, after_sequence: int | None = None
    ) -> tuple[np.ndarray, int]:
        response, payload = self._request(
            {
                "type": "get_image",
                "after_sequence": after_sequence,
                "timeout_sec": float(timeout_sec),
            },
            timeout_sec=timeout_sec + 1.0,
        )
        self._raise_for_error(response)
        if response.get("type") != "image":
            raise RuntimeError(
                f"Expected image response, got {response.get('type')!r}."
            )
        height = int(response["height"])
        width = int(response["width"])
        channels = int(response.get("channels", 3))
        expected_size = height * width * channels
        if len(payload) != expected_size:
            raise ValueError(
                f"Expected {expected_size} RGB bytes, got {len(payload)}."
            )
        frame = np.frombuffer(payload, dtype=np.uint8).reshape(
            height, width, channels
        )
        if channels != 3:
            raise ValueError(f"Expected three RGB channels, got {channels}.")
        return frame.copy(), int(response["image_sequence"])

    def execute(
        self,
        action: NavPrimitive,
        *,
        episode_id: int,
        sequence_id: int,
        timeout_sec: float,
    ) -> PrimitiveResult:
        response, _ = self._request(
            {
                "type": "execute",
                "episode_id": int(episode_id),
                "sequence_id": int(sequence_id),
                "action": int(action),
                "timeout_sec": float(timeout_sec),
            },
            timeout_sec=timeout_sec + 1.0,
        )
        self._raise_for_error(response)
        if response.get("type") != "execute_result":
            raise RuntimeError(
                f"Expected execute_result, got {response.get('type')!r}."
            )
        try:
            status = PrimitiveStatus(int(response["status"]))
        except ValueError:
            status = PrimitiveStatus.CONTROL_ERROR
        return PrimitiveResult(
            status=status,
            message=str(response.get("message", "")),
            distance_m=float(response.get("distance_m", 0.0)),
            yaw_rad=float(response.get("yaw_rad", 0.0)),
        )

    def health(self, *, timeout_sec: float = 2.0) -> dict[str, Any]:
        response, _ = self._request(
            {"type": "health"},
            timeout_sec=timeout_sec,
        )
        self._raise_for_error(response)
        if response.get("type") != "health_result":
            raise RuntimeError(
                f"Expected health_result, got {response.get('type')!r}."
            )
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._heartbeat_stop.set()
        try:
            self._request(
                {"type": "close"},
                timeout_sec=1.0,
                allow_closed=True,
            )
        except Exception:
            pass
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
        self._udp_socket.close()
        if self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)

    def _request(
        self,
        header: dict[str, Any],
        *,
        timeout_sec: float,
        allow_closed: bool = False,
    ) -> tuple[dict[str, Any], bytes]:
        if self._closed and not allow_closed:
            raise RuntimeError("Go2 socket transport is closed.")
        with self._request_lock:
            self._request_id += 1
            request_id = self._request_id
            request = dict(header)
            request["request_id"] = request_id
            previous_timeout = self._socket.gettimeout()
            self._socket.settimeout(timeout_sec)
            try:
                send_frame(self._socket, request)
                response, payload = recv_frame(self._socket)
            except socket.timeout as exc:
                raise TimeoutError(
                    f"Go2 socket request {header.get('type')!r} timed out "
                    f"after {timeout_sec}s."
                ) from exc
            finally:
                self._socket.settimeout(previous_timeout)
            if int(response.get("request_id", -1)) != request_id:
                raise RuntimeError(
                    "Go2 socket response request_id does not match the request."
                )
            return response, payload

    @staticmethod
    def _raise_for_error(response: dict[str, Any]) -> None:
        if response.get("type") == "error":
            raise RuntimeError(str(response.get("message", "Go2 gateway error.")))

    def _heartbeat_loop(self) -> None:
        counter = 0
        destination = (
            self._config.socket_host,
            self._config.socket_udp_heartbeat_port,
        )
        while not self._heartbeat_stop.is_set():
            counter += 1
            heartbeat = {
                "version": PROTOCOL_VERSION,
                "type": "heartbeat",
                "session_id": self._session_id,
                "counter": counter,
                "signature": heartbeat_signature(
                    self._token,
                    self._session_id,
                    counter,
                ),
            }
            try:
                self._udp_socket.sendto(
                    json.dumps(
                        heartbeat,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                    destination,
                )
            except OSError:
                if not self._heartbeat_stop.is_set():
                    return
            self._heartbeat_stop.wait(
                self._config.socket_heartbeat_interval_sec
            )

    @staticmethod
    def _configure_tcp_keepalive(sock: socket.socket) -> None:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for option_name, value in (
            ("TCP_KEEPIDLE", 2),
            ("TCP_KEEPINTVL", 1),
            ("TCP_KEEPCNT", 3),
        ):
            option = getattr(socket, option_name, None)
            if option is not None:
                sock.setsockopt(socket.IPPROTO_TCP, option, value)
