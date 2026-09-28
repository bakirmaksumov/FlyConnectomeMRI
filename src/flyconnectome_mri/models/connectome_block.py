"""Connectome-constrained propagation layer.

BIOLOGY decides WHICH connections exist; TRAINING decides HOW STRONGLY each
existing connection is used.

Mathematically this is ``W_effective = W_trainable * adjacency_mask``, but it is
stored sparsely: there is exactly ONE trainable weight per existing directed
edge and no parameter at all for a non-existent edge. Consequences:
  - a non-existent edge can never carry signal or receive gradient;
  - every graph with the same edge count has exactly the same number of
    parameters (fair comparison between biological and control graphs);
  - no dense N x N matrix is ever built (scales to large graphs).

Signal direction follows the edge direction (presynaptic -> postsynaptic):

    h_new[j] = ReLU( sum over edges i->j of  w_ij * h[i]  +  b_j )

Node j only receives from its in-neighbours. There is no residual/skip path,
so information can move between nodes only along graph edges.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ConnectomeLayer(nn.Module):
    """One round of message passing along fixed directed edges (1 scalar per node).

    Args:
        edges: Array/tensor of shape (E, 2) with [source_index, target_index] rows.
        num_nodes: Number of nodes N (node indices are 0..N-1).
        activation: Apply ReLU after aggregation (default True).
    """

    def __init__(self, edges: np.ndarray | torch.Tensor, num_nodes: int, activation: bool = True) -> None:
        super().__init__()
        edges = torch.as_tensor(np.asarray(edges), dtype=torch.long)
        if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0:
            raise ValueError("edges must have shape (E, 2) with E > 0")
        if edges.min() < 0 or edges.max() >= num_nodes:
            raise ValueError("edge index out of range")
        # Buffers: part of the model state (saved in checkpoints) but NOT trainable.
        self.register_buffer("source", edges[:, 0].clone())
        self.register_buffer("target", edges[:, 1].clone())
        self.num_nodes = num_nodes
        self.activation = activation

        # Fan-in initialisation: each target node sums over its in-degree k,
        # so weights ~ U(-1/sqrt(k), 1/sqrt(k)) keep the output scale stable.
        in_degree = torch.bincount(self.target, minlength=num_nodes).clamp(min=1).float()
        bound = 1.0 / torch.sqrt(in_degree[self.target])
        self.weight = nn.Parameter((torch.rand(len(edges)) * 2 - 1) * bound)  # one weight per EXISTING edge
        self.bias = nn.Parameter(torch.zeros(num_nodes))

    @property
    def num_edges(self) -> int:
        return len(self.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (batch, N) node states -> (batch, N) new node states."""
        if h.ndim != 2 or h.shape[1] != self.num_nodes:
            raise ValueError(f"expected input of shape (batch, {self.num_nodes}), got {tuple(h.shape)}")
        messages = h[:, self.source] * self.weight            # (batch, E): signal along each edge
        out = torch.zeros_like(h).index_add_(1, self.target, messages)  # sum messages per target node
        out = out + self.bias
        return torch.relu(out) if self.activation else out

    def effective_weight_matrix(self) -> torch.Tensor:
        """Dense (N, N) matrix W_eff with W_eff[i, j] = w_ij for edge i->j, else 0 (for inspection/tests)."""
        dense = torch.zeros(self.num_nodes, self.num_nodes, dtype=self.weight.dtype, device=self.weight.device)
        dense[self.source, self.target] = self.weight.detach()
        return dense

    def adjacency_mask(self) -> torch.Tensor:
        """Binary (N, N) mask: 1 where an edge i->j exists."""
        mask = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.bool, device=self.source.device)
        mask[self.source, self.target] = True
        return mask

    def extra_repr(self) -> str:
        return f"num_nodes={self.num_nodes}, num_edges={self.num_edges}, activation={'ReLU' if self.activation else 'none'}"

