from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share_dir = Path(get_package_share_directory("go2_vln_executor"))
    return LaunchDescription(
        [
            Node(
                package="go2_vln_executor",
                executable="go2_vln_executor_node",
                name="go2_vln_executor",
                output="screen",
                parameters=[str(share_dir / "config" / "go2_vln_executor.yaml")],
            )
        ]
    )

