import torch

class Sampler:
    def __init__(self, temperature: float = 1.0, top_k: int = None, top_p: float = None):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        return

    def _apply_temperature(self, probs: torch.FloatTensor) -> torch.FloatTensor:
        if self.temperature != 1.0:
            probs = torch.softmax(torch.log(probs) / self.temperature, dim=-1)
        return probs

    def _apply_top_k(self, probs: torch.FloatTensor) -> torch.FloatTensor:
        if self.top_k is not None:
            values, indices = torch.topk(probs, self.top_k)
            probs = torch.zeros_like(probs).scatter_(-1, indices, values)
            probs = torch.softmax(probs, dim=-1)
        return probs

    def _apply_top_p(self, probs: torch.FloatTensor) -> torch.FloatTensor:
        if self.top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs = probs.scatter(-1, indices_to_remove, 0.0)
            probs = torch.softmax(probs, dim=-1)
        return probs

    def apply(self, probs: torch.FloatTensor) -> int:
        probs = self._apply_temperature(probs)
        # probs = self._apply_top_k(probs)
        # probs = self._apply_top_p(probs)
        return torch.multinomial(probs, num_samples=1).item()