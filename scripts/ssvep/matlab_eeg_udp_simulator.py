#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import socket
import time

import numpy as np


DEFAULT_IP = "127.0.0.1"
DEFAULT_PORT = 5555
DEFAULT_SAMPLE_RATE = 300.0
DEFAULT_WINDOW_SEC = 2.5
DEFAULT_CHANNELS = 21
DEFAULT_INTERVAL_SEC = 0.5
DEFAULT_FREQS = [12.8, 11.2, 8.8]


def build_packet(packet_id: int, eeg: np.ndarray) -> bytes:
    if eeg.ndim != 2:
        raise ValueError("eeg must be 2D, got shape %r" % (eeg.shape,))
    n_channels, n_samples = eeg.shape
    if not (0 <= packet_id <= 0xFFFFFFFF):
        packet_id = packet_id % (0xFFFFFFFF + 1)

    header = bytearray(8)
    header[0:4] = int(packet_id).to_bytes(4, "little", signed=False)
    header[4] = 0
    header[5] = int(n_channels) & 0xFF
    header[6:8] = int(n_samples).to_bytes(2, "little", signed=False)

    # MATLAB sends data_single(:) for a [channels x samples] matrix.
    # That is column-major: sample 0 all channels, sample 1 all channels, ...
    payload = np.asarray(eeg.T, dtype="<f4", order="C").tobytes(order="C")
    return bytes(header) + payload


def generate_eeg(
    packet_id: int,
    target: int,
    mode: str,
    sample_rate: float,
    n_samples: int,
    n_channels: int,
    amplitude: float,
    noise: float,
    freqs: list[float],
    rng: np.random.Generator,
) -> np.ndarray:
    if mode == "zero":
        return np.zeros((n_channels, n_samples), dtype=np.float32)

    t0 = packet_id * DEFAULT_INTERVAL_SEC
    t = t0 + np.arange(n_samples, dtype=np.float64) / sample_rate
    eeg = rng.normal(0.0, noise, size=(n_channels, n_samples))

    if mode == "noise":
        return eeg.astype(np.float32)

    target_index = max(1, min(int(target), len(freqs))) - 1
    freq = float(freqs[target_index])
    base = np.sin(2.0 * math.pi * freq * t)
    harmonic = 0.45 * np.sin(2.0 * math.pi * freq * 2.0 * t)
    ssvep = amplitude * (base + harmonic)

    # Put the strongest synthetic SSVEP response on posterior-like channels,
    # while still adding a weaker response to all channels.
    posterior_channels = [14, 15]
    eeg += ssvep[None, :] * 0.25
    for index in posterior_channels:
        if 0 <= index < n_channels:
            eeg[index, :] += ssvep

    return eeg.astype(np.float32)


def parse_freqs(value: str) -> list[float]:
    freqs = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        freqs.append(float(item))
    if not freqs:
        raise argparse.ArgumentTypeError("at least one frequency is required")
    return freqs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate zzzz_test_move_eeg_broadcast.m UDP EEG packets."
    )
    parser.add_argument("--ip", default=DEFAULT_IP, help="destination IP, same as MATLAB BROADCAST_IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="destination UDP port")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--interval-sec", type=float, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--target", type=int, default=1, choices=[1, 2, 3], help="synthetic SSVEP target")
    parser.add_argument("--cycle-targets", action="store_true", help="cycle target 1/2/3 while sending")
    parser.add_argument("--cycle-every", type=int, default=6, help="packets per target when --cycle-targets is used")
    parser.add_argument("--freqs", type=parse_freqs, default=DEFAULT_FREQS, help="comma-separated target frequencies")
    parser.add_argument("--mode", choices=["ssvep", "noise", "zero"], default="ssvep")
    parser.add_argument("--amplitude", type=float, default=20.0)
    parser.add_argument("--noise", type=float, default=2.0)
    parser.add_argument("--packets", type=int, default=0, help="0 means run forever")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_samples = int(round(float(args.window_sec) * float(args.sample_rate)))
    packet_size = 8 + int(args.channels) * n_samples * 4
    rng = np.random.default_rng(int(args.seed))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    print("正在初始化 UDP 广播...")
    print("✅ UDP 就绪 -> 目标: %s:%d" % (args.ip, args.port))
    print(
        "  [配置] 窗口长度: %.1f秒 | 样本点数: %d | 包大小: %.1fKB"
        % (args.window_sec, n_samples, packet_size / 1024.0)
    )
    print(">>> 模拟 MATLAB EEG 广播启动；Ctrl+C 停止 <<<")

    packet_id = 0
    try:
        while args.packets <= 0 or packet_id < args.packets:
            if args.cycle_targets:
                target = (packet_id // max(1, int(args.cycle_every))) % 3 + 1
            else:
                target = int(args.target)

            eeg = generate_eeg(
                packet_id=packet_id,
                target=target,
                mode=args.mode,
                sample_rate=float(args.sample_rate),
                n_samples=n_samples,
                n_channels=int(args.channels),
                amplitude=float(args.amplitude),
                noise=float(args.noise),
                freqs=list(args.freqs),
                rng=rng,
            )
            packet = build_packet(packet_id, eeg)
            sock.sendto(packet, (args.ip, int(args.port)))
            print(
                "  [📡 成功] ID:%d | Target:%d | Mode:%s | Size:%.1fKB | 维度:[%d,%d] | std:%.4f"
                % (
                    packet_id,
                    target,
                    args.mode,
                    len(packet) / 1024.0,
                    int(args.channels),
                    n_samples,
                    float(np.std(eeg)),
                ),
                flush=True,
            )
            packet_id = (packet_id + 1) % (0xFFFFFFFF + 1)
            time.sleep(max(0.001, float(args.interval_sec)))
    except KeyboardInterrupt:
        print(">>> 模拟广播已停止 <<<")
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
