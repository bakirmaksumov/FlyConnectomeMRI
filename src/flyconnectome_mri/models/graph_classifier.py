"""Connectome-constrained classifier (corrected architecture, "Option A", 2026-09-28).

    MRI (1 x 128 x 128)
      -> CNNEncoder                      (same frozen encoder as the CNN baseline)
      -> 128-dim feature vector
      -> node_norm: BatchNorm1d(128)     (gives every node an input of usable scale)
      -> node initialisation: feature i -> state of node i   (1 scalar per node)
      -> graph_block: ConnectomeLayer x 2 (signal moves only along graph edges)
      -> all 128 final node states       (no pooling)
      -> classifier: Linear(128 -> 1)    (logit; sigmoid = P(glioma))

Why not the first version (mean pooling -> Linear(1 -> 1))? It collapsed: the
image-dependent signal shrank through the sparse layers and was then averaged
into ONE number, so the output was constant (validation AUC exactly 0.5). See
docs/RESEARCH_PLAN.md.

No bypass: the classifier only sees the final node states, and those are
produced only by the graph layers. There is no skip/residual connection, so
every bit of information reaching the prediction has passed through the graph.

Node order: node i is the i-th node of the loaded graph (sorted bodyId for the
biological and rewired graphs, slot i for ER/WS/BA).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from flyconnectome_mri.models.connectome_block import ConnectomeLayer
from flyconnectome_mri.models.encoder import CNNEncoder


class GraphClassifier(nn.Module):
    """CNN encoder + BatchNorm + connectome-constrained graph block + Linear(N -> 1).

    Args:
        edges: (E, 2) directed edges in node-index space.
        num_nodes: Number of graph nodes; must equal ``feature_dim``.
        num_layers: Number of propagation layers.
        in_channels, channels, feature_dim: Encoder settings (as the baseline).
    """

    def __init__(
        self,
        edges: np.ndarray,
        num_nodes: int,
        num_layers: int = 2,
        in_channels: int = 1,
        channels: tuple[int, int] = (16, 32),
        feature_dim: int = 128,
    ) -> None:
        super().__init__()
        if feature_dim != num_nodes:
            raise ValueError(f"feature i -> node i mapping needs feature_dim == num_nodes ({feature_dim} != {num_nodes})")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.encoder = CNNEncoder(in_channels, channels, feature_dim)
        self.node_norm = nn.BatchNorm1d(num_nodes)
        self.graph_block = nn.ModuleList(ConnectomeLayer(edges, num_nodes) for _ in range(num_layers))
        self.classifier = nn.Linear(num_nodes, 1)  # reads ALL final node states

    def node_states(self, x: torch.Tensor) -> torch.Tensor:
        """Final node states (batch, N): the only input of the classifier."""
        h = self.node_norm(self.encoder(x))  # feature i becomes the initial state of node i
        for layer in self.graph_block:
            h = layer(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.node_states(x)).squeeze(-1)

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor) -> dict[str, float]:
        """Signal health on a batch (call in eval mode).

        "std" = standard deviation ACROSS IMAGES per unit, averaged over units:
        how much a unit's value depends on the image. "zero_frac" = share of
        node activations that are exactly 0 after the ReLU; "dead_nodes_frac" =
        share of nodes that are 0 for EVERY image.
        """
        features = self.encoder(x)
        h = self.node_norm(features)
        stats = {"encoder_feature_std": _across_image_std(features)}
        for i, layer in enumerate(self.graph_block, start=1):
            h = layer(h)
            stats[f"graph_layer{i}_std"] = _across_image_std(h)
            stats[f"graph_layer{i}_zero_frac"] = float((h == 0).float().mean())
            stats[f"graph_layer{i}_dead_nodes_frac"] = float((h == 0).all(dim=0).float().mean())
        stats["logit_std"] = float(self.classifier(h).squeeze(-1).std())
        return stats


def _across_image_std(values: torch.Tensor) -> float:
    return float(values.std(dim=0).mean()) if len(values) > 1 else 0.0


def parameter_breakdown(model: nn.Module) -> dict[str, int]:
    """Trainable parameter count per top-level component (and the total)."""
    counts = {name: sum(p.numel() for p in module.parameters() if p.requires_grad)
              for name, module in model.named_children()}
    counts["total"] = sum(counts.values())
    return counts
