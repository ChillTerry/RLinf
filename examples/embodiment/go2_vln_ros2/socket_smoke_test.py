#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rlinf.envs.realworld.go2.go2_vln_env import (
    Go2VLNConfig,
    NavPrimitive,
)
from rlinf.envs.realworld.go2.socket_transport import SocketGo2VLNTransport


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the Go2 socket gateway without loading Uni-NaVid."
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--tcp-port", type=int, default=8765)
    parser.add_argument("--udp-port", type=int, default=8766)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--action",
        choices=["stop", "forward", "left", "right", "no_op"],
        help="Optional primitive to send after receiving one RGB frame.",
    )
    parser.add_argument(
        "--confirm-motion",
        action="store_true",
        help=(
            "Required for forward/left/right. Confirms the robot is in a "
            "clear, supervised test area."
        ),
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=int(time.time_ns() & ((1 << 63) - 1)),
        help="Unique episode ID; defaults to the current time.",
    )
    parser.add_argument("--sequence-id", type=int, default=1)
    parser.add_argument(
        "--save-image",
        help="Optional path for saving the received RGB frame as a PPM file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    moving_actions = {"forward", "left", "right"}
    if args.action in moving_actions and not args.confirm_motion:
        raise RuntimeError(
            f"--action {args.action} can move the robot. Add --confirm-motion "
            "only after checking the estop and clearing the test area."
        )
    config = Go2VLNConfig(
        socket_host=args.host,
        socket_tcp_port=args.tcp_port,
        socket_udp_heartbeat_port=args.udp_port,
        socket_connect_timeout_sec=args.timeout,
    )
    transport = SocketGo2VLNTransport(config)
    try:
        time.sleep(0.5)
        health = transport.health(timeout_sec=args.timeout)
        print(
            "GATEWAY OK: "
            f"udp_heartbeat_fresh={health['udp_heartbeat_fresh']}, "
            f"image_available={health['image_available']}, "
            f"action_active={health['action_active']}, "
            f"image_source={health.get('image_source', 'unknown')}, "
            f"image_topic={health.get('image_topic', 'unknown')}"
        )
        if not health["udp_heartbeat_fresh"]:
            raise RuntimeError(
                "TCP is connected but the gateway did not receive the UDP "
                "heartbeat."
            )
        image, image_sequence = transport.wait_for_image(
            timeout_sec=args.timeout
        )
        print(
            f"RGB OK: sequence={image_sequence}, shape={image.shape}, "
            f"dtype={image.dtype}, min={image.min()}, max={image.max()}"
        )
        if args.save_image:
            output_path = Path(args.save_image).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wb") as output_file:
                output_file.write(
                    f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode(
                        "ascii"
                    )
                )
                output_file.write(image.tobytes(order="C"))
            print(f"RGB saved: {output_path}")
        if args.action:
            action = NavPrimitive[args.action.upper()]
            result = transport.execute(
                action,
                episode_id=args.episode_id,
                sequence_id=args.sequence_id,
                timeout_sec=args.timeout,
            )
            print(
                f"ACTION OK: action={args.action}, status={result.status.name}, "
                f"message={result.message!r}, distance={result.distance_m:.3f}, "
                f"yaw={result.yaw_rad:.3f}"
            )
    finally:
        transport.close()


if __name__ == "__main__":
    main()
