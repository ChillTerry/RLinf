#!/usr/bin/env python3

from __future__ import annotations

import argparse
import statistics
import time

from rlinf.envs.realworld.go2.go2_vln_env import (
    Go2VLNConfig,
    NavPrimitive,
)
from rlinf.envs.realworld.go2.socket_transport import SocketGo2VLNTransport


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure Go2 gateway TCP control, RGB, and STOP latency."
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--tcp-port", type=int, default=8765)
    parser.add_argument("--udp-port", type=int, default=8766)
    parser.add_argument("--control-samples", type=int, default=50)
    parser.add_argument("--image-samples", type=int, default=10)
    parser.add_argument("--stop-samples", type=int, default=0)
    parser.add_argument(
        "--episode-id",
        type=int,
        default=int(time.time_ns() & ((1 << 63) - 1)),
        help="Episode ID for optional STOP samples; defaults to a unique value.",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def print_latency(label: str, values: list[float]) -> None:
    if not values:
        return
    print(
        f"{label}: n={len(values)}, "
        f"mean={statistics.fmean(values):.2f} ms, "
        f"p50={statistics.median(values):.2f} ms, "
        f"p95={percentile(values, 0.95):.2f} ms, "
        f"max={max(values):.2f} ms"
    )


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def main() -> None:
    args = parse_args()
    if min(args.control_samples, args.image_samples, args.stop_samples) < 0:
        raise ValueError("Sample counts cannot be negative.")

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
            "Gateway health: "
            f"udp_heartbeat_fresh={health['udp_heartbeat_fresh']}, "
            f"image_available={health['image_available']}, "
            f"action_active={health['action_active']}"
        )
        if not health["udp_heartbeat_fresh"]:
            raise RuntimeError("UDP heartbeat is not reaching the gateway.")

        control_latencies = []
        for _ in range(args.control_samples):
            start = time.perf_counter()
            transport.health(timeout_sec=args.timeout)
            control_latencies.append(elapsed_ms(start))
        print_latency("TCP control RTT", control_latencies)

        image_latencies = []
        image_mib_per_sec = []
        last_sequence = None
        for _ in range(args.image_samples):
            start = time.perf_counter()
            image, last_sequence = transport.wait_for_image(
                timeout_sec=args.timeout,
                after_sequence=last_sequence,
            )
            duration_ms = elapsed_ms(start)
            image_latencies.append(duration_ms)
            image_mib = image.nbytes / (1024.0 * 1024.0)
            image_mib_per_sec.append(image_mib / (duration_ms / 1000.0))
        print_latency("Fresh RGB request", image_latencies)
        if image_mib_per_sec:
            print(
                "RGB payload throughput: "
                f"mean={statistics.fmean(image_mib_per_sec):.2f} MiB/s, "
                f"min={min(image_mib_per_sec):.2f} MiB/s"
            )

        stop_latencies = []
        for sequence_id in range(1, args.stop_samples + 1):
            start = time.perf_counter()
            result = transport.execute(
                NavPrimitive.STOP,
                episode_id=args.episode_id,
                sequence_id=sequence_id,
                timeout_sec=args.timeout,
            )
            stop_latencies.append(elapsed_ms(start))
            if result.status.name != "SUCCEEDED":
                raise RuntimeError(
                    f"STOP failed at sequence {sequence_id}: "
                    f"{result.status.name} {result.message}"
                )
        print_latency("STOP action RTT", stop_latencies)
    finally:
        transport.close()


if __name__ == "__main__":
    main()
