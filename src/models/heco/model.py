from typing import Dict, List, Optional, Tuple, Union

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from dgl.ops import edge_softmax

from src.models.base import BaseEncoder, BaseRecommender
from src.models.gnn.metrics import InfoNCELoss


class HeCoGATConv(nn.Module):
    """GAT-based convolution layer for network schema view."""

    def __init__(
        self,
        hidden_dim: int,
        attn_drop: float = 0.0,
        negative_slope: float = 0.01,
        activation: Optional[callable] = None,
    ):
        super().__init__()
        self.attn_l = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_r = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.attn_l, gain)
        nn.init.xavier_normal_(self.attn_r, gain)

    def forward(
        self,
        g: dgl.DGLGraph,
        feat_src: torch.Tensor,
        feat_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation."""
        with g.local_scope():
            attn_l = self.attn_drop(self.attn_l)
            attn_r = self.attn_drop(self.attn_r)

            # Compute attention scores
            el = (feat_src * attn_l).sum(dim=-1, keepdim=True)
            er = (feat_dst * attn_r).sum(dim=-1, keepdim=True)
            g.srcdata.update({"ft": feat_src, "el": el})
            g.dstdata["er"] = er
            g.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(g.edata.pop("e"))
            g.edata["a"] = edge_softmax(g, e)

            # Message passing
            g.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            ret = g.dstdata["ft"]
            if self.activation:
                ret = self.activation(ret)
            return ret


class SemanticAttention(nn.Module):
    """Semantic-level attention for both views."""

    def __init__(self, hidden_dim: int, attn_drop: float):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain)
        nn.init.xavier_normal_(self.attn, gain)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Parameters
        ----------
        h : tensor(N, M, d)
            Node embeddings from different meta-paths/types
            N: number of nodes
            M: number of meta-paths/types
            d: hidden dimension

        Returns
        -------
        tensor(N, d)
            Aggregated node embeddings
        """
        attn = self.attn_drop(self.attn)
        w = torch.tanh(self.fc(h)).mean(dim=0).matmul(attn.t())
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((h.shape[0],) + beta.shape)
        return (beta * h).sum(dim=1)


class NetworkSchemaEncoder(nn.Module):
    """Network schema view encoder."""

    def __init__(
        self,
        hidden_dim: int,
        attn_drop: float,
        relations: List[Tuple[str, str, str]],
    ):
        super().__init__()
        self.relations = relations
        self.dtype = relations[0][2]  # Target node type (paper)

        # GAT for each source node type
        self.gats = nn.ModuleDict({r[0]: HeCoGATConv(hidden_dim, attn_drop, activation=F.elu) for r in relations})

        # Semantic attention for combining different relations
        self.attn = SemanticAttention(hidden_dim, attn_drop)

    def forward(
        self,
        g: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward computation."""
        feat_dst = feats[self.dtype][: g.num_dst_nodes(self.dtype)]
        h = []
        for stype, etype, dtype in self.relations:
            h.append(self.gats[stype](g[stype, etype, dtype], feats[stype], feat_dst))
        h = torch.stack(h, dim=1)
        return self.attn(h)


class MetaPathEncoder(nn.Module):
    """Meta-path view encoder."""

    def __init__(
        self,
        num_metapaths: int,
        in_dim: int,
        hidden_dim: int,
        attn_drop: float,
    ):
        super().__init__()
        self.gcns = nn.ModuleList(
            [GraphConv(in_dim, hidden_dim, norm="right", activation=nn.PReLU()) for _ in range(num_metapaths)]
        )
        self.attn = SemanticAttention(hidden_dim, attn_drop)

    def forward(
        self,
        meta_graphs: List[dgl.DGLGraph],
        feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward computation."""
        h = [gcn(mg, feat) for gcn, mg, feat in zip(self.gcns, meta_graphs, feats)]
        h = torch.stack(h, dim=1)
        return self.attn(h)


class HeCoEncoder(BaseEncoder):
    """HeCo encoder combining network schema and meta-path views."""

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int,
        relations: List[Tuple[str, str, str]],
        num_metapaths: int,
        feat_drop: float = 0.3,
        attn_drop: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Feature transformation for each node type
        self.fcs = nn.ModuleDict({ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in in_dims.items()})
        self.feat_drop = nn.Dropout(feat_drop)

        # Two views
        self.network_schema_encoder = NetworkSchemaEncoder(
            hidden_dim=hidden_dim,
            attn_drop=attn_drop,
            relations=relations,
        )
        self.metapath_encoder = MetaPathEncoder(
            num_metapaths=num_metapaths,
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            attn_drop=attn_drop,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        for fc in self.fcs.values():
            nn.init.xavier_normal_(fc.weight, gain)

    def forward(
        self,
        g: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        meta_graphs: Optional[List[dgl.DGLGraph]] = None,
        meta_feats: Optional[List[torch.Tensor]] = None,
        return_all_nodes: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward computation."""
        # Transform features
        h = {ntype: F.elu(self.feat_drop(self.fcs[ntype](feat))) for ntype, feat in feat_dict.items()}

        # Get embeddings from both views
        z_schema = self.network_schema_encoder(g, h)

        if meta_graphs is not None and meta_feats is not None:
            meta_h = [F.elu(self.feat_drop(self.fcs[self.network_schema_encoder.dtype](feat))) for feat in meta_feats]
            z_path = self.metapath_encoder(meta_graphs, meta_h)
        else:
            z_path = z_schema

        return {"schema": z_schema, "path": z_path}


class HeCoRecommender(BaseRecommender):
    """HeCo-based paper recommender."""

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int,
        relations: List[Tuple[str, str, str]],
        num_metapaths: int,
        temperature: float = 0.8,
        lambda_: float = 0.5,
        feat_drop: float = 0.3,
        attn_drop: float = 0.5,
    ):
        encoder = HeCoEncoder(
            in_dims=in_dims,
            hidden_dim=hidden_dim,
            relations=relations,
            num_metapaths=num_metapaths,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
        )
        super().__init__(encoder)

        # Contrastive learning
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim))
        self.criterion = InfoNCELoss(temperature=temperature)
        self.lambda_ = lambda_

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain)

    def forward(
        self,
        g: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        batch_data: Optional[Dict[str, torch.Tensor]] = None,
        meta_graphs: Optional[List[dgl.DGLGraph]] = None,
        meta_feats: Optional[List[torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward computation for both training and inference."""
        embeds = self.encoder(g, feat_dict, meta_graphs, meta_feats)
        z_schema, z_path = embeds["schema"], embeds["path"]

        if batch_data is None:
            # Use meta-path view embeddings for recommendation
            return z_path

        # Project embeddings
        z_schema_proj = self.proj(z_schema)
        z_path_proj = self.proj(z_path)

        # Get positive samples
        pos = batch_data["pos"]  # (batch_size, num_nodes)
        batch_size = pos.shape[0]

        # Compute NCE loss for both directions
        loss_schema = self.criterion(
            z_schema_proj[:batch_size],
            z_path_proj[:batch_size],
            z_path_proj[batch_size:].view(batch_size, -1, self.encoder.hidden_dim),
            pos,
        )
        loss_path = self.criterion(
            z_path_proj[:batch_size],
            z_schema_proj[:batch_size],
            z_schema_proj[batch_size:].view(batch_size, -1, self.encoder.hidden_dim),
            pos,
        )

        loss = self.lambda_ * loss_schema + (1 - self.lambda_) * loss_path
        return {"loss": loss}
