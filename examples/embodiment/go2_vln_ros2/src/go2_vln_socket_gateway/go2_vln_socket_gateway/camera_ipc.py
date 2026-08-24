from __future__ import annotations

import socket

from .protocol import recv_frame, send_frame


def request_camera_image(
    socket_path: str,
    *,
    timeout_sec: float,
) -> bytes:
    """Request one complete encoded image from the local SDK2 sidecar."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout_sec)
    try:
        client.connect(socket_path)
        send_frame(
            client,
            {
                "type": "camera_get_image",
                "request_id": 1,
            },
        )
        response, payload = recv_frame(client)
    finally:
        client.close()

    if response.get("type") != "camera_image":
        raise RuntimeError(
            f"Unexpected camera sidecar response {response.get('type')!r}."
        )
    if not response.get("ok", False):
        raise RuntimeError(str(response.get("message", "SDK2 camera request failed.")))
    if not payload:
        raise RuntimeError("SDK2 camera sidecar returned an empty image.")
    return payload
