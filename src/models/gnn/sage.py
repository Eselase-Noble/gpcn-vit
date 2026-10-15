"""Minimal GraphSAGE-style GNN (plain torch, no torch_geometric dependency).

Message passing is mean-aggregation over neighbours via index_add, so it runs on
any edge_index without extra libraries and degrades gracefully on isolated nodes
(no neighbours -> neighbour term is zero, only the self-transform applies). A mean
readout over nodes gives one graph-level (patient-level) prediction.

Used for the BreakHis feature-similarity graph and, later, the CAMELYON spatial
graph. Kept deliberately small — the point is to isolate the effect of *edges*,
not to chase capacity.
"""
from __future__ import annotations


def build_model(in_dim: int, hidden: int = 128, n_classes: int = 2,
                layers: int = 2, dropout: float = 0.25):
    """Construct the GNN (import torch lazily so the module imports without torch)."""
    import torch
    import torch.nn as nn

    def aggregate(h, edge_index):
        n = h.shape[0]
        if edge_index.numel() == 0:
            return torch.zeros_like(h)
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, h[src])                       # sum neighbours into dst
        deg = torch.zeros(n, device=h.device)
        deg.index_add_(0, dst, torch.ones(src.shape[0], device=h.device))
        return agg / deg.clamp(min=1).unsqueeze(1)           # mean; isolated -> /1 of zeros

    class GraphSAGE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dims = [in_dim] + [hidden] * layers
            self.self_lin = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])
            self.neigh_lin = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden, n_classes)

        def forward(self, x, edge_index):
            """x: (n_nodes, in_dim), edge_index: (2, E) long -> logits (1, n_classes)."""
            h = x
            for sl, nl in zip(self.self_lin, self.neigh_lin):
                h = torch.relu(sl(h) + nl(aggregate(h, edge_index)))
                h = self.dropout(h)
            return self.classifier(h.mean(dim=0, keepdim=True))  # mean readout

    return GraphSAGE()
