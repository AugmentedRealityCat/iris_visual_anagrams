"""Lightweight options shared by the CLI and desktop form."""

from dataclasses import dataclass
import math

QUALITY_METHODS = {"Classic (default)": "plain", "Anagram-MTL · Iris (experimental)": "anagram_mtl",
                   "SyncVC · Iris (experimental)": "syncvc"}


@dataclass
class MethodOptions:
    method: str = "plain"
    interval: int = 5
    iterations: int = 1
    strength: float | None = None
    cpu_offload: bool = True

    def __post_init__(self):
        if self.strength is None:
            self.strength = 0.05

    def validate(self, views=None, *, turbo=False, reduction="mean", steps=None):
        if self.method not in QUALITY_METHODS.values():
            raise ValueError("Unknown quality method")
        if self.method == "plain":
            return self
        if turbo:
            raise ValueError("Anagram-MTL and SyncVC require Standard Iris; turn off Turbo")
        if views is not None and len(views) != 2:
            raise ValueError("Experimental quality methods currently require exactly two views")
        if reduction not in ("auto", "mean"):
            raise ValueError("Experimental quality methods require geometric views and Auto/Mean combining")
        if not 1 <= self.interval <= 100 or not 1 <= self.iterations <= 10:
            raise ValueError("Optimization interval must be 1–100 and passes must be 1–10")
        if steps is not None and self.interval >= steps:
            raise ValueError("Optimization interval must be smaller than the number of generation steps")
        if not math.isfinite(self.strength) or not 0 < self.strength <= 1:
            raise ValueError("Optimization strength must be positive and at most 1")
        return self
