from __future__ import annotations

import socket
import threading

import numpy as np
import pytest

from rlinf.envs.realworld.go2.go2_vln_env import (
    Go2VLNConfig,
    NavPrimitive,
    PrimitiveStatus,
)
from rlinf.envs.realworld.go2.socket_transport import (
    SocketGo2VLNTransport,
    heartbeat_signature,
    recv_frame,
    send_frame,
)


class FakeGateway:
    def __init__(self, token: str, *, mismatch_image_request_id: bool = False):
        self.token = token
        self.mismatch_image_request_id = mismatch_image_request_id
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.bind(("127.0.0.1", 0))
        self.tcp_socket.listen(1)
        self.tcp_port = self.tcp_socket.getsockname()[1]

        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(("127.0.0.1", 0))
        self.udp_socket.settimeout(2.0)
        self.udp_port = self.udp_socket.getsockname()[1]

        self.requests = []
        self.error = None
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def close(self):
        self.thread.join(timeout=2.0)
        self.tcp_socket.close()
        self.udp_socket.close()
        if self.error is not None:
            raise self.error

    def receive_heartbeat(self):
        payload, _ = self.udp_socket.recvfrom(4096)
        import json

        return json.loads(payload.decode("utf-8"))

    def _serve(self):
        try:
            connection, _ = self.tcp_socket.accept()
            with connection:
                while True:
                    request, _ = recv_frame(connection)
                    self.requests.append(request)
                    request_id = request["request_id"]
                    if request["type"] == "hello":
                        send_frame(
                            connection,
                            {
                                "type": "hello_result",
                                "request_id": request_id,
                                "ok": request["token"] == self.token,
                            },
                        )
                    elif request["type"] == "get_image":
                        image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(
                            3, 4, 3
                        )
                        send_frame(
                            connection,
                            {
                                "type": "image",
                                "request_id": (
                                    request_id + 1
                                    if self.mismatch_image_request_id
                                    else request_id
                                ),
                                "image_sequence": 7,
                                "height": 3,
                                "width": 4,
                                "channels": 3,
                            },
                            image.tobytes(),
                        )
                    elif request["type"] == "execute":
                        send_frame(
                            connection,
                            {
                                "type": "execute_result",
                                "request_id": request_id,
                                "status": int(PrimitiveStatus.SUCCEEDED),
                                "message": "done",
                                "distance_m": 0.25,
                                "yaw_rad": 0.0,
                            },
                        )
                    elif request["type"] == "health":
                        send_frame(
                            connection,
                            {
                                "type": "health_result",
                                "request_id": request_id,
                                "udp_heartbeat_fresh": True,
                                "image_available": True,
                                "image_sequence": 7,
                                "action_active": False,
                            },
                        )
                    elif request["type"] == "close":
                        send_frame(
                            connection,
                            {
                                "type": "close_result",
                                "request_id": request_id,
                            },
                        )
                        return
        except Exception as error:
            self.error = error


def test_socket_transport_exchanges_image_action_and_udp_heartbeat(monkeypatch):
    token = "test-secret"
    monkeypatch.setenv("GO2_VLN_TOKEN", token)
    gateway = FakeGateway(token)
    config = Go2VLNConfig(
        socket_host="127.0.0.1",
        socket_tcp_port=gateway.tcp_port,
        socket_udp_heartbeat_port=gateway.udp_port,
        socket_heartbeat_interval_sec=0.02,
    )

    transport = SocketGo2VLNTransport(config)
    try:
        frame, sequence = transport.wait_for_image(timeout_sec=1.0)
        health = transport.health(timeout_sec=1.0)
        result = transport.execute(
            NavPrimitive.FORWARD,
            episode_id=3,
            sequence_id=4,
            timeout_sec=1.0,
        )
        heartbeat = gateway.receive_heartbeat()
    finally:
        transport.close()
        gateway.close()

    assert frame.shape == (3, 4, 3)
    assert sequence == 7
    assert result.status == PrimitiveStatus.SUCCEEDED
    assert health["udp_heartbeat_fresh"]
    assert result.distance_m == pytest.approx(0.25)
    assert heartbeat["session_id"]
    assert heartbeat["signature"] == heartbeat_signature(
        token,
        heartbeat["session_id"],
        heartbeat["counter"],
    )
    assert [request["type"] for request in gateway.requests] == [
        "hello",
        "get_image",
        "health",
        "execute",
        "close",
    ]


def test_socket_transport_requires_configured_secret(monkeypatch):
    monkeypatch.delenv("GO2_VLN_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="GO2_VLN_TOKEN"):
        SocketGo2VLNTransport(Go2VLNConfig(socket_host="127.0.0.1"))


def test_protocol_rejects_mismatched_response_id(monkeypatch):
    token = "test-secret"
    monkeypatch.setenv("GO2_VLN_TOKEN", token)
    gateway = FakeGateway(token, mismatch_image_request_id=True)
    config = Go2VLNConfig(
        socket_host="127.0.0.1",
        socket_tcp_port=gateway.tcp_port,
        socket_udp_heartbeat_port=gateway.udp_port,
    )
    transport = SocketGo2VLNTransport(config)
    try:
        with pytest.raises(RuntimeError, match="request_id"):
            transport.wait_for_image(timeout_sec=1.0)
    finally:
        transport.close()
        gateway.close()
