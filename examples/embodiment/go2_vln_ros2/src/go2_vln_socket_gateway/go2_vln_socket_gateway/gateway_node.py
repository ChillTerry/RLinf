from __future__ import annotations

import hmac
import json
import math
import os
import socket
import threading
import time
from typing import Any

import numpy as np
import rclpy
from go2_vln_interfaces.action import ExecuteNavPrimitive
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import UInt64

from .camera_ipc import request_camera_image
from .protocol import (
    PROTOCOL_VERSION,
    heartbeat_signature,
    recv_frame,
    send_frame,
)


class Go2VlnSocketGateway(Node):
    def __init__(self) -> None:
        super().__init__("go2_vln_socket_gateway")

        self._bind_host = self.declare_parameter("bind_host", "0.0.0.0").value
        self._tcp_port = int(self.declare_parameter("tcp_port", 8765).value)
        self._udp_port = int(self.declare_parameter("udp_heartbeat_port", 8766).value)
        self._image_source = self.declare_parameter(
            "image_source", "ros_topic"
        ).value
        self._camera_socket_path = self.declare_parameter(
            "camera_socket_path", "/tmp/go2_vln_camera.sock"
        ).value
        self._camera_request_timeout_sec = float(
            self.declare_parameter("camera_request_timeout_sec", 4.0).value
        )
        self._image_topic = self.declare_parameter(
            "image_topic", "/camera/camera/color/image_raw"
        ).value
        self._image_message_type = self.declare_parameter(
            "image_message_type", "sensor_msgs_image"
        ).value
        self._go2_video_resolution = self.declare_parameter(
            "go2_video_resolution", "720p"
        ).value
        self._output_image_width = int(
            self.declare_parameter("output_image_width", 640).value
        )
        self._output_image_height = int(
            self.declare_parameter("output_image_height", 480).value
        )
        self._action_name = self.declare_parameter(
            "action_name", "/go2_vln/execute_nav_primitive"
        ).value
        self._heartbeat_topic = self.declare_parameter(
            "heartbeat_topic", "/go2_vln/heartbeat"
        ).value
        self._auth_token_env = self.declare_parameter(
            "auth_token_env", "GO2_VLN_TOKEN"
        ).value
        self._require_auth = bool(self.declare_parameter("require_auth", True).value)
        self._action_server_timeout_sec = float(
            self.declare_parameter("action_server_timeout_sec", 5.0).value
        )
        self._max_action_result_timeout_sec = float(
            self.declare_parameter("max_action_result_timeout_sec", 20.0).value
        )
        self._udp_heartbeat_timeout_sec = float(
            self.declare_parameter("udp_heartbeat_timeout_sec", 1.0).value
        )

        self._token = os.environ.get(self._auth_token_env, "")
        if self._require_auth and not self._token:
            raise RuntimeError(
                f"{self._auth_token_env} must be set when require_auth=true."
            )

        self._image_condition = threading.Condition()
        self._image: np.ndarray | None = None
        self._image_sequence = 0
        self._camera_capture_lock = threading.Lock()
        self._camera_prime_thread: threading.Thread | None = None

        self._session_lock = threading.Lock()
        self._session_id: str | None = None
        self._session_ip: str | None = None
        self._last_heartbeat_time = 0.0
        self._last_heartbeat_counter = 0

        self._goal_lock = threading.Lock()
        self._active_goal = None
        self._active_goal_cancel_requested = False

        self._shutdown_event = threading.Event()
        self._tcp_socket = self._create_tcp_server_socket()
        self._udp_socket = self._create_udp_server_socket()
        self._client_threads: list[threading.Thread] = []

        self._image_subscription = None
        if self._image_source == "unitree_video_client":
            if not os.path.isabs(self._camera_socket_path):
                raise ValueError("camera_socket_path must be absolute.")
            if self._camera_request_timeout_sec <= 0:
                raise ValueError("camera_request_timeout_sec must be positive.")
        elif (
            self._image_source == "ros_topic"
            and self._image_message_type == "sensor_msgs_image"
        ):
            self._image_subscription = self.create_subscription(
                Image,
                self._image_topic,
                self._on_image,
                qos_profile_sensor_data,
            )
        elif (
            self._image_source == "ros_topic"
            and self._image_message_type == "go2_front_video"
        ):
            from unitree_go.msg import Go2FrontVideoData

            if self._go2_video_resolution not in {"720p", "360p", "180p"}:
                raise ValueError("go2_video_resolution must be 720p, 360p, or 180p.")
            self._image_subscription = self.create_subscription(
                Go2FrontVideoData,
                self._image_topic,
                self._on_go2_front_video,
                qos_profile_sensor_data,
            )
        else:
            raise ValueError(
                "image_source must be 'unitree_video_client' or "
                "'ros_topic'; ros_topic requires image_message_type "
                "'go2_front_video' or 'sensor_msgs_image'."
            )
        self._action_client = ActionClient(
            self,
            ExecuteNavPrimitive,
            self._action_name,
        )
        self._heartbeat_publisher = self.create_publisher(
            UInt64,
            self._heartbeat_topic,
            10,
        )
        self._heartbeat_publish_counter = 0
        self._heartbeat_timer = self.create_timer(
            0.2,
            self._heartbeat_watchdog,
        )

        self._tcp_thread = threading.Thread(
            target=self._tcp_server_loop,
            name="go2-vln-tcp-server",
            daemon=True,
        )
        self._udp_thread = threading.Thread(
            target=self._udp_server_loop,
            name="go2-vln-udp-heartbeat",
            daemon=True,
        )
        self._tcp_thread.start()
        self._udp_thread.start()

        if self._image_source == "unitree_video_client":
            self._camera_prime_thread = threading.Thread(
                target=self._prime_sidecar_camera,
                name="go2-vln-camera-prime",
                daemon=True,
            )
            self._camera_prime_thread.start()

        self.get_logger().info(
            f"Go2 socket gateway listening on TCP {self._bind_host}:"
            f"{self._tcp_port} and UDP {self._bind_host}:{self._udp_port}"
        )
        if self._image_source == "ros_topic":
            self.get_logger().info(
                f"RGB source is ROS topic {self._image_topic!r} "
                f"({self._image_message_type})."
            )

    def wait_for_action_server(self) -> bool:
        available = self._action_client.wait_for_server(
            timeout_sec=self._action_server_timeout_sec
        )
        if not available:
            self.get_logger().error(
                f"Action server {self._action_name!r} was not available "
                f"within {self._action_server_timeout_sec}s."
            )
        return available

    def close(self) -> None:
        if self._shutdown_event.is_set():
            return
        self._shutdown_event.set()
        self._cancel_active_goal()
        for server_socket in (self._tcp_socket, self._udp_socket):
            if server_socket is not None:
                try:
                    server_socket.close()
                except OSError:
                    pass
        for thread in (self._tcp_thread, self._udp_thread):
            if thread.is_alive():
                thread.join(timeout=2.0)
        if (
            self._camera_prime_thread is not None
            and self._camera_prime_thread.is_alive()
        ):
            self._camera_prime_thread.join(timeout=2.0)
        for thread in self._client_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)

    def _tcp_server_loop(self) -> None:
        server = self._tcp_socket
        while not self._shutdown_event.is_set():
            try:
                connection, address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._configure_tcp_keepalive(connection)
            thread = threading.Thread(
                target=self._serve_client,
                args=(connection, address),
                name=f"go2-vln-client-{address[0]}",
                daemon=True,
            )
            self._client_threads.append(thread)
            thread.start()

    def _serve_client(
        self,
        connection: socket.socket,
        address: tuple[str, int],
    ) -> None:
        authenticated_session: str | None = None
        try:
            hello, _ = recv_frame(connection)
            request_id = int(hello.get("request_id", -1))
            if hello.get("type") != "hello":
                send_frame(
                    connection,
                    {
                        "type": "hello_result",
                        "request_id": request_id,
                        "ok": False,
                        "message": "First request must be hello.",
                    },
                )
                return
            session_id = str(hello.get("session_id", ""))
            token = str(hello.get("token", ""))
            if not session_id or (
                self._require_auth and not hmac.compare_digest(token, self._token)
            ):
                send_frame(
                    connection,
                    {
                        "type": "hello_result",
                        "request_id": request_id,
                        "ok": False,
                        "message": "Authentication failed.",
                    },
                )
                return
            if not self._claim_session(session_id, address[0]):
                send_frame(
                    connection,
                    {
                        "type": "hello_result",
                        "request_id": request_id,
                        "ok": False,
                        "message": "Another RLinf client is already connected.",
                    },
                )
                return
            authenticated_session = session_id
            send_frame(
                connection,
                {
                    "type": "hello_result",
                    "request_id": request_id,
                    "ok": True,
                    "message": "Connected.",
                },
            )

            while not self._shutdown_event.is_set():
                request, _ = recv_frame(connection)
                request_id = int(request.get("request_id", -1))
                request_type = request.get("type")
                try:
                    if request_type == "get_image":
                        response, payload = self._handle_get_image(
                            request,
                            request_id,
                        )
                        send_frame(connection, response, payload)
                    elif request_type == "execute":
                        send_frame(
                            connection,
                            self._handle_execute(request, request_id),
                        )
                    elif request_type == "health":
                        with self._image_condition:
                            image_available = self._image is not None
                            image_sequence = self._image_sequence
                        with self._goal_lock:
                            action_active = self._active_goal is not None
                        send_frame(
                            connection,
                            {
                                "type": "health_result",
                                "request_id": request_id,
                                "udp_heartbeat_fresh": (self._heartbeat_is_fresh()),
                                "image_available": image_available,
                                "image_sequence": image_sequence,
                                "image_source": self._image_source,
                                "image_topic": self._image_topic,
                                "action_active": action_active,
                            },
                        )
                    elif request_type == "close":
                        send_frame(
                            connection,
                            {
                                "type": "close_result",
                                "request_id": request_id,
                            },
                        )
                        return
                    else:
                        raise ValueError(f"Unsupported request type {request_type!r}.")
                except Exception as error:
                    send_frame(
                        connection,
                        {
                            "type": "error",
                            "request_id": request_id,
                            "message": str(error),
                        },
                    )
        except (ConnectionError, OSError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning(
                f"Socket client {address[0]} disconnected: {error}"
            )
        finally:
            try:
                connection.close()
            except OSError:
                pass
            if authenticated_session is not None:
                self._release_session(authenticated_session)

    def _handle_get_image(
        self,
        request: dict[str, Any],
        request_id: int,
    ) -> tuple[dict[str, Any], bytes]:
        after_sequence_value = request.get("after_sequence")
        after_sequence = (
            None if after_sequence_value is None else int(after_sequence_value)
        )
        timeout_sec = self._bounded_timeout(
            request.get("timeout_sec", 5.0),
            maximum=30.0,
        )
        if self._image_source == "unitree_video_client":
            self._capture_sidecar_image(
                min(timeout_sec, self._camera_request_timeout_sec)
            )
        deadline = time.monotonic() + timeout_sec
        with self._image_condition:
            if self._image_source == "ros_topic":
                # A streaming camera may have advanced well beyond the sequence
                # consumed by RLinf while the robot was moving. Wait for a frame
                # published after this request, rather than returning a cached
                # frame captured during the primitive.
                after_sequence = max(
                    -1 if after_sequence is None else after_sequence,
                    self._image_sequence,
                )
            while self._image is None or (
                after_sequence is not None and self._image_sequence <= after_sequence
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"No fresh RGB image arrived within {timeout_sec}s."
                    )
                self._image_condition.wait(timeout=remaining)
            image = self._image.copy()
            image_sequence = self._image_sequence
        payload = image.tobytes(order="C")
        self.get_logger().info(
            "Socket RGB response: "
            f"sequence={image_sequence} bytes={int(image.nbytes)} "
            f"shape={int(image.shape[1])}x{int(image.shape[0])}x"
            f"{int(image.shape[2])}"
        )
        return (
            {
                "type": "image",
                "request_id": request_id,
                "image_sequence": image_sequence,
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "channels": int(image.shape[2]),
                "encoding": "rgb8",
            },
            payload,
        )

    def _handle_execute(
        self,
        request: dict[str, Any],
        request_id: int,
    ) -> dict[str, Any]:
        action = int(request["action"])
        if (
            action < ExecuteNavPrimitive.Goal.STOP
            or action > ExecuteNavPrimitive.Goal.NO_OP
        ):
            raise ValueError(f"Unsupported navigation action {action}.")
        action_names = ("STOP", "FORWARD", "LEFT", "RIGHT", "NO_OP")
        self.get_logger().info(
            "Socket command received: "
            f"episode={int(request['episode_id'])} "
            f"sequence={int(request['sequence_id'])} "
            f"action={action_names[action]}({action})"
        )
        if (
            action
            not in (
                ExecuteNavPrimitive.Goal.STOP,
                ExecuteNavPrimitive.Goal.NO_OP,
            )
            and not self._heartbeat_is_fresh()
        ):
            return {
                "type": "execute_result",
                "request_id": request_id,
                "status": int(ExecuteNavPrimitive.Result.CONTROL_ERROR),
                "message": "No fresh authenticated UDP heartbeat.",
                "distance_m": 0.0,
                "yaw_rad": 0.0,
            }

        timeout_sec = self._bounded_timeout(
            request.get("timeout_sec", 15.0),
            maximum=self._max_action_result_timeout_sec,
        )
        goal = ExecuteNavPrimitive.Goal()
        goal.episode_id = int(request["episode_id"])
        goal.sequence_id = int(request["sequence_id"])
        goal.action = action

        goal_future = self._action_client.send_goal_async(goal)
        goal_handle = self._wait_future(goal_future, timeout_sec)
        if not goal_handle.accepted:
            return {
                "type": "execute_result",
                "request_id": request_id,
                "status": int(ExecuteNavPrimitive.Result.REJECTED),
                "message": "Local Go2 action server rejected the goal.",
                "distance_m": 0.0,
                "yaw_rad": 0.0,
            }
        with self._goal_lock:
            self._active_goal = goal_handle
            self._active_goal_cancel_requested = False
        try:
            wrapped_result = self._wait_future(
                goal_handle.get_result_async(),
                timeout_sec,
            )
            result = wrapped_result.result
            self.get_logger().info(
                "Socket command completed: "
                f"episode={int(request['episode_id'])} "
                f"sequence={int(request['sequence_id'])} "
                f"status={int(result.status)} "
                f"distance={float(result.distance_m):.3f} "
                f"yaw={float(result.yaw_rad):.3f}"
            )
            return {
                "type": "execute_result",
                "request_id": request_id,
                "status": int(result.status),
                "message": str(result.message),
                "distance_m": float(result.distance_m),
                "yaw_rad": float(result.yaw_rad),
            }
        except TimeoutError:
            goal_handle.cancel_goal_async()
            return {
                "type": "execute_result",
                "request_id": request_id,
                "status": int(ExecuteNavPrimitive.Result.TIMEOUT),
                "message": f"Local action result timed out after {timeout_sec}s.",
                "distance_m": 0.0,
                "yaw_rad": 0.0,
            }
        finally:
            with self._goal_lock:
                if self._active_goal is goal_handle:
                    self._active_goal = None
                    self._active_goal_cancel_requested = False

    def _udp_server_loop(self) -> None:
        server = self._udp_socket
        while not self._shutdown_event.is_set():
            try:
                payload, address = server.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                heartbeat = json.loads(payload.decode("utf-8"))
                self._accept_heartbeat(heartbeat, address[0])
            except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
                continue

    def _accept_heartbeat(
        self,
        heartbeat: dict[str, Any],
        source_ip: str,
    ) -> None:
        if (
            int(heartbeat.get("version", -1)) != PROTOCOL_VERSION
            or heartbeat.get("type") != "heartbeat"
        ):
            return
        session_id = str(heartbeat.get("session_id", ""))
        counter = int(heartbeat.get("counter", -1))
        signature = str(heartbeat.get("signature", ""))
        expected_signature = heartbeat_signature(
            self._token,
            session_id,
            counter,
        )
        with self._session_lock:
            if (
                session_id != self._session_id
                or source_ip != self._session_ip
                or counter <= self._last_heartbeat_counter
                or not hmac.compare_digest(signature, expected_signature)
            ):
                return
            self._last_heartbeat_counter = counter
            self._last_heartbeat_time = time.monotonic()

    def _heartbeat_watchdog(self) -> None:
        if self._heartbeat_is_fresh():
            self._heartbeat_publish_counter += 1
            message = UInt64()
            message.data = self._heartbeat_publish_counter
            self._heartbeat_publisher.publish(message)
        else:
            self._cancel_active_goal()

    def _heartbeat_is_fresh(self) -> bool:
        with self._session_lock:
            return (
                self._session_id is not None
                and self._last_heartbeat_time > 0.0
                and time.monotonic() - self._last_heartbeat_time
                <= self._udp_heartbeat_timeout_sec
            )

    def _claim_session(self, session_id: str, source_ip: str) -> bool:
        with self._session_lock:
            if self._session_id is not None:
                return False
            self._session_id = session_id
            self._session_ip = source_ip
            self._last_heartbeat_time = 0.0
            self._last_heartbeat_counter = 0
            return True

    def _release_session(self, session_id: str) -> None:
        with self._session_lock:
            if self._session_id != session_id:
                return
            self._session_id = None
            self._session_ip = None
            self._last_heartbeat_time = 0.0
            self._last_heartbeat_counter = 0
        self._cancel_active_goal()

    def _cancel_active_goal(self) -> None:
        with self._goal_lock:
            if self._active_goal is None or self._active_goal_cancel_requested:
                return
            self._active_goal_cancel_requested = True
            self._active_goal.cancel_goal_async()

    def _on_image(self, message: Image) -> None:
        try:
            image = self._normalize_image_size(self._decode_image(message))
        except (TypeError, ValueError) as error:
            self.get_logger().error(f"Invalid RGB image: {error}")
            return
        self._store_image(image)

    def _prime_sidecar_camera(self) -> None:
        warning_logged = False
        while not self._shutdown_event.is_set():
            try:
                self._capture_sidecar_image(self._camera_request_timeout_sec)
                self.get_logger().info("SDK2 camera sidecar is ready.")
                return
            except (OSError, RuntimeError, ValueError) as error:
                if not warning_logged:
                    self.get_logger().warning(
                        f"Waiting for SDK2 camera sidecar: {error}"
                    )
                    warning_logged = True
                self._shutdown_event.wait(0.5)

    def _capture_sidecar_image(self, timeout_sec: float) -> None:
        with self._camera_capture_lock:
            encoded = request_camera_image(
                self._camera_socket_path,
                timeout_sec=timeout_sec,
            )
            try:
                import cv2

                compressed = np.frombuffer(encoded, dtype=np.uint8)
                bgr_image = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
                if bgr_image is None:
                    raise ValueError("OpenCV failed to decode the SDK2 JPEG image.")
                image = self._normalize_image_size(bgr_image[:, :, ::-1])
            except (ImportError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Invalid image from SDK2 camera sidecar: {error}"
                ) from error
            self._store_image(image)

    def _on_go2_front_video(self, message) -> None:
        encoded = getattr(message, f"video{self._go2_video_resolution}")
        if not encoded:
            self.get_logger().warning(
                f"Empty Go2 {self._go2_video_resolution} video frame."
            )
            return
        try:
            import cv2

            compressed = np.asarray(encoded, dtype=np.uint8)
            bgr_image = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
            if bgr_image is None:
                raise ValueError("OpenCV failed to decode the compressed frame.")
            image = self._normalize_image_size(bgr_image[:, :, ::-1])
        except (ImportError, TypeError, ValueError) as error:
            self.get_logger().error(f"Invalid Go2 front video frame: {error}")
            return
        self._store_image(image)

    def _store_image(self, image: np.ndarray) -> None:
        with self._image_condition:
            self._image = np.ascontiguousarray(image)
            self._image_sequence += 1
            self._image_condition.notify_all()

    def _normalize_image_size(self, image: np.ndarray) -> np.ndarray:
        target_width = self._output_image_width
        target_height = self._output_image_height
        source_height, source_width = image.shape[:2]
        if (source_width, source_height) == (target_width, target_height):
            return np.ascontiguousarray(image)

        import cv2

        target_aspect = target_width / target_height
        source_aspect = source_width / source_height
        if source_aspect > target_aspect:
            crop_width = max(1, round(source_height * target_aspect))
            start_x = (source_width - crop_width) // 2
            image = image[:, start_x:start_x + crop_width]
        elif source_aspect < target_aspect:
            crop_height = max(1, round(source_width / target_aspect))
            start_y = (source_height - crop_height) // 2
            image = image[start_y:start_y + crop_height, :]
        resized = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        return np.ascontiguousarray(resized)

    @staticmethod
    def _decode_image(message: Image) -> np.ndarray:
        encodings = {
            "rgb8": (3, False),
            "bgr8": (3, True),
            "rgba8": (4, False),
            "bgra8": (4, True),
            "mono8": (1, False),
        }
        encoding = str(message.encoding).lower()
        if encoding not in encodings:
            raise ValueError(
                f"Unsupported encoding {message.encoding!r}; expected one of "
                f"{sorted(encodings)}."
            )
        channels, swap_rb = encodings[encoding]
        row_bytes = int(message.width) * channels
        if int(message.step) < row_bytes:
            raise ValueError("Image step is smaller than encoded row width.")
        data = np.frombuffer(message.data, dtype=np.uint8)
        expected_size = int(message.height) * int(message.step)
        if data.size < expected_size:
            raise ValueError("Image buffer is shorter than height * step.")
        image = data[:expected_size].reshape(int(message.height), int(message.step))
        image = image[:, :row_bytes].reshape(
            int(message.height),
            int(message.width),
            channels,
        )
        if channels == 1:
            image = np.repeat(image, 3, axis=2)
        else:
            image = image[:, :, :3]
        if swap_rb:
            image = image[:, :, ::-1]
        return np.ascontiguousarray(image)

    @staticmethod
    def _wait_future(future, timeout_sec: float):
        completed = threading.Event()
        future.add_done_callback(lambda _: completed.set())
        if not completed.wait(timeout=timeout_sec):
            raise TimeoutError(f"ROS2 future timed out after {timeout_sec}s.")
        exception = future.exception()
        if exception is not None:
            raise exception
        return future.result()

    @staticmethod
    def _bounded_timeout(value: Any, *, maximum: float) -> float:
        timeout = float(value)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"Timeout must be finite and positive, got {value!r}.")
        return min(timeout, maximum)

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

    def _create_tcp_server_socket(self) -> socket.socket:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._bind_host, self._tcp_port))
        server.listen(4)
        server.settimeout(0.5)
        return server

    def _create_udp_server_socket(self) -> socket.socket:
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._bind_host, self._udp_port))
        server.settimeout(0.5)
        return server


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = Go2VlnSocketGateway()
        if not node.wait_for_action_server():
            raise RuntimeError("Local Go2 action server is unavailable.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()
