#!/usr/bin/env python3
"""Robot-local RealSense ROS2 publisher with automatic reconnect."""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class RealSensePublisher(Node):
    def __init__(self) -> None:
        super().__init__("go2_vln_realsense_publisher")

        self._width = int(self.declare_parameter("width", 640).value)
        self._height = int(self.declare_parameter("height", 480).value)
        self._fps = int(self.declare_parameter("fps", 30).value)
        self._enable_color = bool(
            self.declare_parameter("enable_color", True).value
        )
        self._enable_depth = bool(
            self.declare_parameter("enable_depth", False).value
        )
        self._align_depth = bool(
            self.declare_parameter("align_depth_to_color", True).value
        )
        self._serial_no = str(self.declare_parameter("serial_no", "").value)
        self._reconnect_interval_sec = float(
            self.declare_parameter("reconnect_interval_sec", 2.0).value
        )
        self._frame_timeout_ms = int(
            self.declare_parameter("frame_timeout_ms", 1000).value
        )
        self._color_topic = str(
            self.declare_parameter(
                "color_topic", "/camera/camera/color/image_raw"
            ).value
        )
        self._color_info_topic = str(
            self.declare_parameter(
                "color_info_topic", "/camera/camera/color/camera_info"
            ).value
        )
        self._depth_topic = str(
            self.declare_parameter(
                "depth_topic",
                "/camera/camera/aligned_depth_to_color/image_raw",
            ).value
        )
        self._depth_info_topic = str(
            self.declare_parameter(
                "depth_info_topic",
                "/camera/camera/aligned_depth_to_color/camera_info",
            ).value
        )
        self._color_frame_id = str(
            self.declare_parameter(
                "color_frame_id", "camera_color_optical_frame"
            ).value
        )
        self._depth_frame_id = str(
            self.declare_parameter(
                "depth_frame_id", "camera_color_optical_frame"
            ).value
        )

        if self._width <= 0 or self._height <= 0 or self._fps <= 0:
            raise ValueError("width, height, and fps must be positive.")
        if not self._enable_color and not self._enable_depth:
            raise ValueError("At least one RealSense stream must be enabled.")
        if self._reconnect_interval_sec <= 0 or self._frame_timeout_ms <= 0:
            raise ValueError(
                "reconnect_interval_sec and frame_timeout_ms must be positive."
            )

        self._color_publisher = (
            self.create_publisher(Image, self._color_topic, qos_profile_sensor_data)
            if self._enable_color
            else None
        )
        self._color_info_publisher = (
            self.create_publisher(
                CameraInfo,
                self._color_info_topic,
                qos_profile_sensor_data,
            )
            if self._enable_color
            else None
        )
        self._depth_publisher = (
            self.create_publisher(Image, self._depth_topic, qos_profile_sensor_data)
            if self._enable_depth
            else None
        )
        self._depth_info_publisher = (
            self.create_publisher(
                CameraInfo,
                self._depth_info_topic,
                qos_profile_sensor_data,
            )
            if self._enable_depth
            else None
        )

        self._rs = None
        self._pipeline = None
        self._align = None
        self._color_intrinsics = None
        self._depth_intrinsics = None
        self._connected = False
        self._last_reconnect_attempt = float("-inf")

        self._timer = self.create_timer(1.0 / self._fps, self._on_timer)
        self.get_logger().info(
            "RealSense publisher initialized: "
            f"{self._width}x{self._height}@{self._fps}fps, "
            f"color={self._enable_color}, depth={self._enable_depth}."
        )

    def _load_realsense(self):
        if self._rs is None:
            try:
                import pyrealsense2 as rs
            except ImportError as error:
                raise RuntimeError(
                    "pyrealsense2 is not installed in the Python environment "
                    "used by this ROS2 node."
                ) from error
            self._rs = rs
        return self._rs

    def _start_camera(self) -> bool:
        try:
            rs = self._load_realsense()
            pipeline = rs.pipeline()
            config = rs.config()
            if self._serial_no:
                config.enable_device(self._serial_no)
            if self._enable_color:
                config.enable_stream(
                    rs.stream.color,
                    self._width,
                    self._height,
                    rs.format.bgr8,
                    self._fps,
                )
            if self._enable_depth:
                config.enable_stream(
                    rs.stream.depth,
                    self._width,
                    self._height,
                    rs.format.z16,
                    self._fps,
                )

            profile = pipeline.start(config)
            color_intrinsics = None
            depth_intrinsics = None
            if self._enable_color:
                color_intrinsics = (
                    profile.get_stream(rs.stream.color)
                    .as_video_stream_profile()
                    .get_intrinsics()
                )
            if self._enable_depth:
                depth_intrinsics = (
                    profile.get_stream(rs.stream.depth)
                    .as_video_stream_profile()
                    .get_intrinsics()
                )

            self._pipeline = pipeline
            self._color_intrinsics = color_intrinsics
            self._depth_intrinsics = depth_intrinsics
            self._align = (
                rs.align(rs.stream.color)
                if self._align_depth
                and self._enable_color
                and self._enable_depth
                else None
            )
            self._connected = True
            self.get_logger().info(
                "RealSense connected successfully; publishing color on "
                f"{self._color_topic!r}."
            )
            return True
        except Exception as error:
            self.get_logger().warning(f"RealSense connection failed: {error}")
            self._stop_camera(log=False)
            return False

    def _stop_camera(self, *, log: bool = True) -> None:
        self._connected = False
        pipeline = self._pipeline
        self._pipeline = None
        self._align = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if log:
            self.get_logger().warning("RealSense disconnected/stopped.")

    def _on_timer(self) -> None:
        if not self._connected:
            current_time = time.monotonic()
            if (
                current_time - self._last_reconnect_attempt
                >= self._reconnect_interval_sec
            ):
                self._last_reconnect_attempt = current_time
                self._start_camera()
            return

        try:
            frames = self._pipeline.wait_for_frames(
                timeout_ms=self._frame_timeout_ms
            )
            frames = self._align.process(frames) if self._align is not None else frames
            stamp = self.get_clock().now().to_msg()

            if self._enable_color:
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    self._color_publisher.publish(
                        self._image_message(
                            color_image,
                            encoding="bgr8",
                            frame_id=self._color_frame_id,
                            stamp=stamp,
                        )
                    )
                    self._color_info_publisher.publish(
                        self._camera_info_message(
                            self._color_intrinsics,
                            frame_id=self._color_frame_id,
                            stamp=stamp,
                        )
                    )

            if self._enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    self._depth_publisher.publish(
                        self._image_message(
                            depth_image,
                            encoding="16UC1",
                            frame_id=self._depth_frame_id,
                            stamp=stamp,
                        )
                    )
                    depth_intrinsics = (
                        self._color_intrinsics
                        if self._align is not None
                        else self._depth_intrinsics
                    )
                    self._depth_info_publisher.publish(
                        self._camera_info_message(
                            depth_intrinsics,
                            frame_id=self._depth_frame_id,
                            stamp=stamp,
                        )
                    )
        except Exception as error:
            self.get_logger().error(f"RealSense frame retrieval failed: {error}")
            self._stop_camera()

    @staticmethod
    def _image_message(
        image: np.ndarray,
        *,
        encoding: str,
        frame_id: str,
        stamp,
    ) -> Image:
        image = np.ascontiguousarray(image)
        message = Image()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.height = int(image.shape[0])
        message.width = int(image.shape[1])
        message.encoding = encoding
        message.is_bigendian = False
        message.step = int(image.strides[0])
        message.data = image.tobytes(order="C")
        return message

    @staticmethod
    def _camera_info_message(intrinsics, *, frame_id: str, stamp) -> CameraInfo:
        message = CameraInfo()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.width = int(intrinsics.width)
        message.height = int(intrinsics.height)
        message.distortion_model = "plumb_bob"
        message.d = [float(value) for value in intrinsics.coeffs]
        message.k = [
            float(intrinsics.fx),
            0.0,
            float(intrinsics.ppx),
            0.0,
            float(intrinsics.fy),
            float(intrinsics.ppy),
            0.0,
            0.0,
            1.0,
        ]
        message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        message.p = [
            float(intrinsics.fx),
            0.0,
            float(intrinsics.ppx),
            0.0,
            0.0,
            float(intrinsics.fy),
            float(intrinsics.ppy),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return message

    def destroy_node(self):
        self._stop_camera(log=False)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = RealSensePublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
