from glob import glob

from setuptools import find_packages, setup

package_name = "go2_vln_socket_gateway"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RLinf Authors",
    maintainer_email="rlinf@example.com",
    description="Socket gateway for RLinf Go2 VLN.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "go2_vln_camera_sidecar = go2_vln_socket_gateway.camera_sidecar:main",
            "go2_vln_realsense_publisher = go2_vln_socket_gateway.realsense_publisher:main",
            "go2_vln_socket_gateway = go2_vln_socket_gateway.gateway_node:main",
        ],
    },
)
