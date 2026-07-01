#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import queue
import socket
import threading
import time

import numpy as np


HEADER_BYTES = 8


@dataclass(frozen=True)
class EEGPacket:
    packet_id: int
    reserved: int
    n_channels: int
    n_samples: int
    eeg: np.ndarray
    received_at: float
    dropped_packets: int = 0


def parse_eeg_packet(
    buf: bytes,
    expected_channels: int = 21,
    expected_samples: int = 750,
) -> EEGPacket:
    if len(buf) < HEADER_BYTES:
        raise ValueError("packet too short: %d bytes" % len(buf))

    packet_id = int.from_bytes(buf[0:4], "little", signed=False)
    reserved = buf[4]
    n_channels = int(buf[5])
    n_samples = int.from_bytes(buf[6:8], "little", signed=False)
    expected_len = HEADER_BYTES + n_channels * n_samples * 4
    if len(buf) != expected_len:
        raise ValueError("invalid packet length: got %d, expected %d" % (len(buf), expected_len))
    if expected_channels and n_channels != expected_channels:
        raise ValueError("invalid channel count: got %d, expected %d" % (n_channels, expected_channels))
    if expected_samples and n_samples != expected_samples:
        raise ValueError("invalid sample count: got %d, expected %d" % (n_samples, expected_samples))

    values = np.frombuffer(buf[HEADER_BYTES:], dtype="<f4")
    eeg = values.reshape((n_samples, n_channels)).T.copy()
    if not np.all(np.isfinite(eeg)):
        raise ValueError("packet contains NaN or Inf")

    return EEGPacket(
        packet_id=packet_id,
        reserved=reserved,
        n_channels=n_channels,
        n_samples=n_samples,
        eeg=eeg,
        received_at=time.monotonic(),
    )


class EEGUDPReceiver:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5555,
        expected_channels: int = 21,
        expected_samples: int = 750,
        queue_size: int = 4,
    ):
        self.host = host
        self.port = int(port)
        self.expected_channels = int(expected_channels)
        self.expected_samples = int(expected_samples)
        self.queue: queue.Queue[EEGPacket] = queue.Queue(maxsize=max(1, int(queue_size)))
        self.last_error = ""
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._last_packet_id: int | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="eeg-udp-receiver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_latest(self) -> EEGPacket | None:
        latest = None
        while True:
            try:
                latest = self.queue.get_nowait()
            except queue.Empty:
                return latest

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock = sock
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.settimeout(0.2)
            while not self._stop_event.is_set():
                try:
                    buf, _addr = sock.recvfrom(1024 * 1024)
                    packet = parse_eeg_packet(buf, self.expected_channels, self.expected_samples)
                    dropped = self._count_dropped(packet.packet_id)
                    if dropped:
                        packet = EEGPacket(
                            packet_id=packet.packet_id,
                            reserved=packet.reserved,
                            n_channels=packet.n_channels,
                            n_samples=packet.n_samples,
                            eeg=packet.eeg,
                            received_at=packet.received_at,
                            dropped_packets=dropped,
                        )
                    self._put_latest(packet)
                except socket.timeout:
                    continue
                except OSError:
                    if not self._stop_event.is_set():
                        self.last_error = "socket closed unexpectedly"
                    break
                except Exception as exc:
                    self.last_error = str(exc)
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def _count_dropped(self, packet_id: int) -> int:
        dropped = 0
        if self._last_packet_id is not None and packet_id > self._last_packet_id + 1:
            dropped = packet_id - self._last_packet_id - 1
        self._last_packet_id = packet_id
        return dropped

    def _put_latest(self, packet: EEGPacket) -> None:
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(packet)


def describe_packet(packet: EEGPacket) -> str:
    eeg = packet.eeg
    return (
        "packet_id=%d channels=%d samples=%d mean=%.4f std=%.4f min=%.4f max=%.4f dropped=%d"
        % (
            packet.packet_id,
            packet.n_channels,
            packet.n_samples,
            float(np.mean(eeg)),
            float(np.std(eeg)),
            float(np.min(eeg)),
            float(np.max(eeg)),
            packet.dropped_packets,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Receive MATLAB EEG UDP packets and print statistics.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--channels", type=int, default=21)
    parser.add_argument("--samples", type=int, default=750)
    args = parser.parse_args()

    receiver = EEGUDPReceiver(args.host, args.port, args.channels, args.samples)
    receiver.start()
    print("listening on %s:%d" % (args.host, args.port), flush=True)
    try:
        while True:
            packet = receiver.get_latest()
            if packet:
                print(describe_packet(packet), flush=True)
            elif receiver.last_error:
                print("receiver warning: %s" % receiver.last_error, flush=True)
                receiver.last_error = ""
            time.sleep(0.05)
    except KeyboardInterrupt:
        receiver.stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
