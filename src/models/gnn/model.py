from typing import Dict, Optional, Union

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseEncoder, BaseRecommender
from src.models.gnn.layers import HeteroConv
from src.models.gnn.metrics import InfoNCELoss


class HGTEncoder(BaseEncoder):
    """Heterogeneous Graph Transformer encoder."""

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.2,
        residual: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual

        # Input projection for each node type
        self.projectors = nn.ModuleDict(
            {ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in in_dims.items()}
        )

        # GNN layers
        self.layers = nn.ModuleList(
            [
                HeteroConv(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer normalization and dropout
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        for projector in self.projectors.values():
            nn.init.xavier_normal_(projector.weight)
            if projector.bias is not None:
                nn.init.zeros_(projector.bias)

    def forward(
        self,
        g: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        return_all_nodes: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass to generate node embeddings."""
        # Project input features
        h_dict = {
            ntype: self.drop(F.relu(self.projectors[ntype](feat)))
            for ntype, feat in feat_dict.items()
        }

        # GNN layers
        for i in range(self.num_layers):
            h_new = {}

            # Process each edge type
            for etype in g.canonical_etypes:
                h_new[etype[2]] = self.layers[i](g, h_dict, etype)

            # Residual connection and normalization
            for ntype in h_dict:
                if self.residual and h_dict[ntype].shape == h_new[ntype].shape:
                    h_new[ntype] = h_new[ntype] + h_dict[ntype]
                h_new[ntype] = self.norms[i](h_new[ntype])

            h_dict = h_new

        return h_dict if return_all_nodes else h_dict["paper"]


class HGTRecommender(BaseRecommender):
    """Heterogeneous Graph Transformer for paper recommendation."""

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        temperature: float = 0.1,
        dropout: float = 0.2,
    ):
        # Create encoder
        encoder = HGTEncoder(
            in_dims=in_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        super().__init__(encoder)

        # Contrastive learning loss
        self.criterion = InfoNCELoss(temperature=temperature)

    def forward(
        self,
        g: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        batch_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for both training and inference."""
        # Get paper embeddings
        paper_embeds = self.encoder(g, feat_dict)

        # If in inference mode, return embeddings
        if batch_data is None:
            return paper_embeds

        # Get batch embeddings
        anchor_embeds = paper_embeds[batch_data["anchor"]]
        pos_embeds = paper_embeds[batch_data["positive"]]
        neg_embeds = paper_embeds[batch_data["negative"].view(-1)].view(
            len(anchor_embeds), -1, self.encoder.hidden_dim
        )

        # Compute loss
        loss = self.criterion(anchor_embeds, pos_embeds, neg_embeds)

        return {"loss": loss}
