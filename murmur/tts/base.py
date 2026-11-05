"""Common interface for text-to-speech engines."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EngineUnavailable(RuntimeError):
    """The engine can't run on this machine. The message tells the user how to fix it."""


class TTSEngine(ABC):
    name = "base"
    sample_rate = 24000

    @abstractmethod
    def synthesize(self, text: str, *, speed: float = 1.0, intensity: float = 0.5) -> np.ndarray:
        """Return mono float32 samples in [-1, 1] at ``self.sample_rate``.

        ``speed`` is relative to the engine's base speed (1.0 = unchanged). ``intensity`` runs 0..1 and is
        used by engines that can vary expressiveness; the others ignore it.
        """

    def close(self) -> None:  # pragma: no cover - optional hook
        pass
