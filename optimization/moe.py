# optimization/moe.py
"""
Sparse MoE model definitions for MNIST.
Clean, importable, ready for loading .pth weights.
Aligned with 02_moe.ipynb notebook logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Sparse MoE Components
# -----------------------------
class NoisyTopKRouter(nn.Module):
    def __init__(self, input_dim, num_experts, k=1, noise_std=1.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_experts)
        self.k = k
        self.noise_std = noise_std

    def forward(self, x):
        logits = self.linear(x)
        if self.training:
            logits += torch.randn_like(logits) * self.noise_std

        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)
        gates = F.softmax(topk_vals, dim=-1)

        sparse_gates = torch.zeros_like(logits)
        sparse_gates.scatter_(1, topk_idx, gates)

        return sparse_gates, topk_idx


class Expert(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):  # ← CRITICAL: Add this
        return self.net(x)


class SparseMoELayer(nn.Module):
    def __init__(self, dim, num_experts=4, k=1, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor

        self.router = NoisyTopKRouter(dim, num_experts, k)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])

    def forward(self, x):
        gates, topk_idx = self.router(x)
        batch_size = x.size(0)
        capacity = int(self.capacity_factor * batch_size / self.num_experts)
        output = torch.zeros_like(x)
        load = torch.zeros(self.num_experts, device=x.device)

        active_experts = topk_idx.unique()
        for expert_id in active_experts:
            mask = gates[:, expert_id] > 0
            tokens = x[mask][:capacity]
            weights = gates[mask, expert_id][:tokens.size(0)].unsqueeze(1)

            if tokens.size(0) == 0:
                continue

            expert_out = self.experts[expert_id](tokens)
            output[mask][:tokens.size(0)] += expert_out * weights
            load[expert_id] += tokens.size(0)

        load_dist = load / (load.sum() + 1e-8)
        load_loss = - (load_dist * torch.log(load_dist + 1e-8)).sum()

        return output, load_loss, load_dist


class FFN_MoE(nn.Module):
    """
    Feed-forward MNIST model with a sparse MoE layer.
    """
    def __init__(self, input_dim=28*28, hidden_dim=256, num_experts=4, k=1, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.moe = SparseMoELayer(hidden_dim, num_experts=num_experts, k=k)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        moe_out, load_loss, load_dist = self.moe(x)
        x = x + moe_out
        logits = self.fc_out(x)
        return logits, load_loss, load_dist
