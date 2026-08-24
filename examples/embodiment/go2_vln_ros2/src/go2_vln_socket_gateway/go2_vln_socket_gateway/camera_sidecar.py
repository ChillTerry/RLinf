from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import stat
import threading
from collections.abc import Callable

from .protocol import recv_frame, send_frame

LOGGER = logging.getLogger("go2_vln_camera_sidecar")


class CameraSidecarServer:
    """Local Unix-socket server around SDK2 VideoClient.GetImageSample()."""

    def __init__(
        self,
        socket_path: str,
        capture: Callable[[], tuple[int, object]],
    ) -> None:
        if not os.path.isabs(socket_path):
            raise ValueError("Camera sidecar socket path must be absolute.")
        self._socket_path = socket_path
        self._capture = capture
        self._stop_event = threading.Event()
        self._server: socket.socket | None = None

    def serve(self) -> None:
        self._remove_stale_socket()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server = server
        server.bind(self._socket_path)
        os.chmod(self._socket_path, 0o600)
        server.listen(2)
        server.settimeout(0.5)
        LOGGER.info("Listening on Unix socket %s", self._socket_path)

        try:
            while not self._stop_event.is_set():
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        break
                    raise
                with connection:
                    self._serve_connection(connection)
        finally:
            self.close()

    def close(self) -> None:
        self._stop_event.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        self._remove_owned_socket()

    def _serve_connection(self, connection: socket.socket) -> None:
        request_id = -1
        try:
            request, _ = recv_frame(connection)
            request_id = int(request.get("request_id", -1))
            if request.get("type") != "camera_get_image":
                raise ValueError(f"Unsupported camera request {request.get('type')!r}.")

            code, data = self._capture()
            if int(code) != 0:
                send_frame(
                    connection,
                    {
                        "type": "camera_image",
                        "request_id": request_id,
                        "ok": False,
                        "code": int(code),
                        "message": f"VideoClient returned code {int(code)}.",
                    },
                )
                return

            payload = bytes(data)
            if not payload:
                raise RuntimeError("VideoClient returned an empty image.")
            send_frame(
                connection,
                {
                    "type": "camera_image",
                    "request_id": request_id,
                    "ok": True,
                    "code": 0,
                    "encoding": "jpeg",
                },
                payload,
            )
            LOGGER.info("Returned JPEG image: %d bytes", len(payload))
        except Exception as error:
            LOGGER.error("Camera request failed: %s", error)
            try:
                send_frame(
                    connection,
                    {
                        "type": "camera_image",
                        "request_id": request_id,
                        "ok": False,
                        "message": str(error),
                    },
                )
            except OSError:
                pass

    def _remove_stale_socket(self) -> None:
        try:
            mode = os.stat(self._socket_path).st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(mode):
            raise RuntimeError(
                f"Refusing to replace non-socket path {self._socket_path}."
            )
        os.unlink(self._socket_path)

    def _remove_owned_socket(self) -> None:
        try:
            mode = os.stat(self._socket_path).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(mode):
            os.unlink(self._socket_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve complete Go2 JPEG frames from SDK2 over localhost."
    )
    parser.add_argument(
        "--socket-path",
        default="/tmp/go2_vln_camera.sock",
    )
    parser.add_argument(
        "--network-interface",
        default=os.environ.get("GO2_NETWORK_INTERFACE", ""),
        help="SDK2 DDS network interface; empty uses the SDK default.",
    )
    parser.add_argument("--request-timeout", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] [go2_vln_camera_sidecar] %(message)s",
    )
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive.")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient

    if args.network_interface:
        LOGGER.info(
            "Initializing SDK2 on network interface %s",
            args.network_interface,
        )
        ChannelFactoryInitialize(0, args.network_interface)
    else:
        LOGGER.info("Initializing SDK2 with its default network interface")
        ChannelFactoryInitialize(0)

    client = VideoClient()
    client.SetTimeout(float(args.request_timeout))
    client.Init()

    server = CameraSidecarServer(args.socket_path, client.GetImageSample)

    def stop_server(_signum, _frame) -> None:
        server.close()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    server.serve()


if __name__ == "__main__":
    main()
