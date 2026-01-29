# optimization/moe.py
"""
Sparse MoE model definitions for MNIST.
Clean, importable, ready for loading .pth weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Router for Sparse MoE
# -----------------------------
class NoisyTopKRouter(nn.Module):
    """
    Top-k router with optional noise for token assignment to experts.
    """
    def __init__(self, input_dim: int, num_experts: int, k: int = 2, noise_std: float = 1.0, temperature: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_experts)
        self.k = k
        self.noise_std = noise_std
        self.temperature = temperature

    def forward(self, x: torch.Tensor):
        logits = self.linear(x) / self.temperature

        if self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-k gating
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)
        gates = F.softmax(topk_vals, dim=-1)

        sparse_gates = torch.zeros_like(logits)
        sparse_gates.scatter_(1, topk_idx, gates)

        # Router confidence: average max probability
        router_confidence = topk_vals.max(dim=-1).values.mean()
        return sparse_gates, topk_idx, router_confidence


# -----------------------------
# Expert network
# -----------------------------
class Expert(nn.Module):
    """
    Simple feed-forward expert.
    """
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


# -----------------------------
# Sparse Mixture-of-Experts layer
# -----------------------------
class SparseMoELayer(nn.Module):
    """
    Sparse MoE layer with top-k routing and capacity limit.
    """
    def __init__(self, input_dim: int, num_experts: int = 8, k: int = 2, capacity_factor: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.router = NoisyTopKRouter(input_dim, num_experts, k=k)
        self.experts = nn.ModuleList([Expert(input_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
        gates, topk_idx, confidence = self.router(x)
        batch_size = x.size(0)
        expert_outputs = torch.zeros_like(x)
        load = torch.zeros(self.num_experts, device=x.device)

        for expert_id in range(self.num_experts):
            mask = gates[:, expert_id] > 0
            tokens = x[mask]

            if tokens.shape[0] == 0:
                continue

            # Capacity limiting
            cap = int(self.capacity_factor * (batch_size / self.num_experts))
            if tokens.shape[0] > cap:
                tokens = tokens[:cap]

            load[expert_id] = tokens.shape[0]

            out = self.experts[expert_id](tokens)
            expert_outputs[mask][:out.shape[0]] += out

        # Load-balancing loss (entropy)
        load_dist = load / (load.sum() + 1e-8)
        load_loss = (load_dist * torch.log(load_dist + 1e-8)).sum()

        return expert_outputs, load_loss, load_dist, confidence


# -----------------------------
# Full FFN MoE model
# -----------------------------
class FFN_MoE(nn.Module):
    """
    Feed-forward MNIST model with a sparse MoE layer.
    """
    def __init__(self, input_dim: int = 28*28, hidden_dim: int = 256, num_experts: int = 8, k: int = 2, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.moe = SparseMoELayer(hidden_dim, num_experts=num_experts, k=k)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        moe_out, load_loss, load_dist, confidence = self.moe(x)
        x = x + moe_out  # residual

        logits = self.fc_out(x)
        return logits, load_loss, load_dist, confidence


# -----------------------------
# Optional: test import
# -----------------------------
if __name__ == "__main__":
    print("moe.py imported successfully")
