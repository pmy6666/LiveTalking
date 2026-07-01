#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class SSVEPResult:
    target_index: int | None
    confidence: float
    scores: list[float]


class FFTSSVEPClassifier:
    def __init__(self, sample_rate: float = 300.0, harmonics: int = 2, channels: list[int] | None = None):
        self.sample_rate = float(sample_rate)
        self.harmonics = max(1, int(harmonics))
        self.channels = channels or []

    def predict(self, eeg: np.ndarray, frequencies: list[float]) -> SSVEPResult:
        if eeg.ndim != 2:
            raise ValueError("eeg must be 2D, got shape %r" % (eeg.shape,))
        valid_freqs = [float(freq) for freq in frequencies if float(freq) > 0]
        if not valid_freqs:
            return SSVEPResult(None, 0.0, [])

        signal = self._select_channels(eeg).astype(np.float64, copy=False)
        signal = signal - np.mean(signal, axis=1, keepdims=True)
        window = np.hanning(signal.shape[1])
        spectrum = np.fft.rfft(signal * window[None, :], axis=1)
        power = np.abs(spectrum) ** 2
        bins = np.fft.rfftfreq(signal.shape[1], d=1.0 / self.sample_rate)

        scores = []
        for freq in valid_freqs:
            score = 0.0
            for harmonic in range(1, self.harmonics + 1):
                target_freq = freq * harmonic
                if target_freq >= bins[-1]:
                    continue
                bin_index = int(np.argmin(np.abs(bins - target_freq)))
                lo = max(0, bin_index - 1)
                hi = min(power.shape[1], bin_index + 2)
                score += float(np.mean(power[:, lo:hi]))
            scores.append(score)

        total = float(sum(scores))
        if total <= 0:
            return SSVEPResult(None, 0.0, scores)
        best_index = int(np.argmax(scores))
        sorted_scores = sorted(scores, reverse=True)
        second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        confidence = (scores[best_index] - second) / total
        return SSVEPResult(best_index + 1, float(confidence), scores)

    def _select_channels(self, eeg: np.ndarray) -> np.ndarray:
        if not self.channels:
            return eeg
        indexes = [index for index in self.channels if 0 <= index < eeg.shape[0]]
        return eeg[indexes] if indexes else eeg


class DecisionSmoother:
    def __init__(
        self,
        decision_windows: int = 3,
        min_votes: int = 2,
        confidence_threshold: float = 0.2,
        submit_cooldown_sec: float = 2.0,
    ):
        self.decision_windows = max(1, int(decision_windows))
        self.min_votes = max(1, int(min_votes))
        self.confidence_threshold = float(confidence_threshold)
        self.submit_cooldown_sec = float(submit_cooldown_sec)
        self.votes: deque[int] = deque(maxlen=self.decision_windows)
        self.last_submit_at = 0.0

    def reset(self) -> None:
        self.votes.clear()

    def update(self, result: SSVEPResult, now: float | None = None) -> int | None:
        now = time.monotonic() if now is None else now
        if result.target_index is None or result.confidence < self.confidence_threshold:
            return None
        self.votes.append(int(result.target_index))
        winner, count = Counter(self.votes).most_common(1)[0]
        if count < self.min_votes:
            return None
        if now - self.last_submit_at < self.submit_cooldown_sec:
            return None
        self.last_submit_at = now
        self.votes.clear()
        return int(winner)
