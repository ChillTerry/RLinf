import os
from pathlib import Path

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    executor_share = Path(get_package_share_directory("go2_vln_executor"))
    gateway_share = Path(get_package_share_directory("go2_vln_socket_gateway"))
    gateway_prefix = Path(get_package_prefix("go2_vln_socket_gateway"))
    sidecar_executable = (
        gateway_prefix / "lib" / "go2_vln_socket_gateway" / "go2_vln_camera_sidecar"
    )
    sidecar_command = [
        str(sidecar_executable),
        "--socket-path",
        "/tmp/go2_vln_camera.sock",
        "--request-timeout",
        "3.0",
    ]
    network_interface = os.environ.get("GO2_NETWORK_INTERFACE", "")
    if network_interface:
        sidecar_command.extend(["--network-interface", network_interface])
    image_source = LaunchConfiguration("image_source")
    image_topic = LaunchConfiguration("image_topic")
    image_message_type = LaunchConfiguration("image_message_type")
    start_realsense_publisher = LaunchConfiguration(
        "start_realsense_publisher"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "image_source",
                default_value="ros_topic",
                description=(
                    "Use ros_topic for RealSense or "
                    "unitree_video_client for SDK2."
                ),
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/camera/camera/color/image_raw",
                description="RealSense sensor_msgs/Image color topic.",
            ),
            DeclareLaunchArgument(
                "image_message_type",
                default_value="sensor_msgs_image",
                description="Message adapter used by the socket gateway.",
            ),
            DeclareLaunchArgument(
                "start_realsense_publisher",
                default_value="true",
                description=(
                    "Start the bundled pyrealsense2 ROS2 publisher. Set false "
                    "when an external camera driver already owns the device."
                ),
            ),
            Node(
                package="go2_vln_executor",
                executable="go2_vln_executor_node",
                name="go2_vln_executor",
                output="screen",
                parameters=[str(executor_share / "config" / "go2_vln_executor.yaml")],
            ),
            ExecuteProcess(
                cmd=sidecar_command,
                name="go2_vln_camera_sidecar",
                output="screen",
                condition=IfCondition(
                    PythonExpression(
                        ["'", image_source, "' == 'unitree_video_client'"]
                    )
                ),
                respawn=True,
                respawn_delay=2.0,
            ),
            Node(
                package="go2_vln_socket_gateway",
                executable="go2_vln_realsense_publisher",
                name="go2_vln_realsense_publisher",
                output="screen",
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            image_source,
                            "' == 'ros_topic' and '",
                            start_realsense_publisher,
                            "' == 'true'",
                        ]
                    )
                ),
                respawn=True,
                respawn_delay=2.0,
                parameters=[
                    str(gateway_share / "config" / "go2_vln_realsense.yaml"),
                    {"color_topic": image_topic},
                ],
            ),
            Node(
                package="go2_vln_socket_gateway",
                executable="go2_vln_socket_gateway",
                name="go2_vln_socket_gateway",
                output="screen",
                parameters=[
                    str(gateway_share / "config" / "go2_vln_socket_gateway.yaml"),
                    {
                        "image_source": image_source,
                        "image_topic": image_topic,
                        "image_message_type": image_message_type,
                    },
                ],
            ),
        ]
    )
