from typing import Dict, Tuple

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import edge_softmax


class RelationAttention(nn.Module):
    """Relation-aware attention layer for heterogeneous graphs.

    This layer implements a multi-head attention mechanism that:
    1. Is aware of different relation types in the graph
    2. Can attend differently to different types of neighbors
    3. Preserves relation semantics during message passing

    Based on ideas from HGT and HAN architectures.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.2,
        negative_slope: float = 0.2,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.sqrt_dk = torch.sqrt(torch.tensor(self.d_k))

        # Linear transformations for Query, Key, Value
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)

        # Relation-specific transformations
        self.relation_att = nn.Parameter(torch.Tensor(num_heads, self.d_k, self.d_k))

        # Output transformation
        self.out_trans = nn.Linear(out_dim, out_dim)

        # Dropout
        self.drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.q.weight, gain=gain)
        nn.init.xavier_normal_(self.k.weight, gain=gain)
        nn.init.xavier_normal_(self.v.weight, gain=gain)
        nn.init.xavier_normal_(self.out_trans.weight, gain=gain)
        nn.init.xavier_normal_(self.relation_att, gain=gain)

    def forward(
        self,
        g: dgl.DGLGraph,
        feat_src: torch.Tensor,
        feat_dst: torch.Tensor,
        etype: Tuple[str, str, str],
    ) -> torch.Tensor:
        """
        Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The input graph
        feat_src : torch.Tensor
            Source node features (N_src, in_dim)
        feat_dst : torch.Tensor
            Destination node features (N_dst, in_dim)
        etype : tuple of str
            The edge type (src_type, edge_type, dst_type)

        Returns
        -------
        torch.Tensor
            Output node features (N_dst, out_dim)
        """
        with g.local_scope():
            # Linear transformations
            q = self.q(feat_dst).view(-1, self.num_heads, self.d_k)
            k = self.k(feat_src).view(-1, self.num_heads, self.d_k)
            v = self.v(feat_src).view(-1, self.num_heads, self.d_k)

            # Compute attention scores
            # Relation-aware attention
            k = torch.einsum("bnd,hde->bhne", k, self.relation_att)
            attn = (q.unsqueeze(-2) @ k.transpose(-2, -1)) / self.sqrt_dk
            attn = self.leaky_relu(attn)

            # Apply attention dropout and softmax
            attn = self.drop(edge_softmax(g, attn))

            # Aggregate messages
            g.srcdata["v"] = v
            g.edata["a"] = attn
            g.update_all(fn.u_mul_e("v", "a", "m"), fn.sum("m", "h"))
            out = g.dstdata["h"]

            # Output transformation
            out = out.view(-1, self.out_dim)
            out = self.drop(self.out_trans(out))

            return out


class HeteroConv(nn.Module):
    """Heterogeneous Graph Convolution Layer.

    This layer applies relation-specific transformations and aggregates
    messages from different relation types.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # Relation-specific attention layers
        self.rel_attn = RelationAttention(in_dim=in_dim, out_dim=out_dim, num_heads=num_heads, dropout=dropout)

        # Output transformation after aggregating all relations
        self.out_trans = nn.Linear(out_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.out_trans.weight, gain=gain)

    def forward(self, g: dgl.DGLGraph, feat_dict: Dict[str, torch.Tensor], etype: Tuple[str, str, str]) -> torch.Tensor:
        """
        Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The input heterogeneous graph
        feat_dict : dict[str, torch.Tensor]
            Node feature dictionary
        etype : tuple of str
            Edge type to process

        Returns
        -------
        torch.Tensor
            Output node features
        """
        src_type, _, dst_type = etype
        feat_src = feat_dict[src_type]
        feat_dst = feat_dict[dst_type]

        # Apply relation-specific attention
        rel_out = self.rel_attn(g, feat_src, feat_dst, etype)

        # Final transformation
        out = self.drop(F.relu(self.out_trans(rel_out)))

        return out
