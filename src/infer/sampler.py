import torch
import random
import numpy as np

class Sampler:
    def __init__(self, temperature: float = 1.0, top_k: int = None, top_p: float = None):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        return

    def _apply_temperature(self, probs: torch.FloatTensor) -> torch.FloatTensor:
        if self.temperature != 1.0:
            probs = [p ** (1.0 / self.temperature) for p in probs]
            probs = [p / sum(probs) for p in probs]
        return probs

    def _apply_top_k(self, probs: torch.FloatTensor) -> torch.FloatTensor:
        if self.top_k is not None:
            top_k_indices = np.argsort(probs)[-self.top_k:]
            probs = [p if i in top_k_indices else 0 for i, p in enumerate(probs)]
            probs = [p / sum(probs) for p in probs]
        return probs

    def _apply_top_p(self, probs: torch.FloatTensor) -> torch.FloatTensor:
        if self.top_p is not None:
            sorted_probs = np.sort(probs)[::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = sorted_probs[np.where(cumulative_probs >= self.top_p)[0][0]]
            probs = [p if p >= cutoff else 0 for p in probs]
            probs = [p / sum(probs) for p in probs]
        return probs

    def apply(self, probs: torch.FloatTensor) -> int:
        probs = self._apply_temperature(probs)
        probs = self._apply_top_k(probs)
        probs = self._apply_top_p(probs)

        return random.choices(range(len(probs)), weights=probs, k=1)[0]