from typing import Dict, Optional, Union

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import edge_softmax

from src.models.base import BaseEncoder, BaseRecommender
from src.models.gnn.metrics import InfoNCELoss


class MicroConv(nn.Module):
    """Micro-level convolution for a single relation type."""

    def __init__(
        self,
        out_dim: int,
        num_heads: int,
        fc_src: nn.Linear,
        fc_dst: nn.Linear,
        attn_src: nn.Parameter,
        feat_drop: float = 0.0,
        negative_slope: float = 0.2,
        activation: Optional[callable] = None,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.fc_src = fc_src
        self.fc_dst = fc_dst
        self.attn_src = attn_src
        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation

    def forward(self, g: dgl.DGLGraph, feat_src: torch.Tensor, feat_dst: torch.Tensor) -> torch.Tensor:
        """Forward computation for micro-level convolution."""
        with g.local_scope():
            feat_src = self.fc_src(self.feat_drop(feat_src)).view(-1, self.num_heads, self.out_dim)
            feat_dst = self.fc_dst(self.feat_drop(feat_dst)).view(-1, self.num_heads, self.out_dim)

            # Compute attention scores
            el = (feat_src * self.attn_src[:, : self.out_dim]).sum(dim=-1, keepdim=True)
            er = (feat_dst * self.attn_src[:, self.out_dim :]).sum(dim=-1, keepdim=True)
            g.srcdata.update({"ft": feat_src, "el": el})
            g.dstdata["er"] = er
            g.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(g.edata.pop("e"))
            g.edata["a"] = edge_softmax(g, e)

            # Message passing
            g.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            ret = g.dstdata["ft"].view(-1, self.num_heads * self.out_dim)
            if self.activation:
                ret = self.activation(ret)
            return ret


class MacroConv(nn.Module):
    """Macro-level convolution across different relation types."""

    def __init__(
        self,
        out_dim: int,
        num_heads: int,
        fc_node: Dict[str, nn.Linear],
        fc_rel: Dict[str, nn.Linear],
        attn: nn.Parameter,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.fc_node = fc_node
        self.fc_rel = fc_rel
        self.attn = attn
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        node_feats: Dict[str, torch.Tensor],
        rel_feats: Dict[tuple, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward computation for macro-level convolution."""
        # Transform node and relation features
        node_feats = {
            ntype: self.fc_node[ntype](feat).view(-1, self.num_heads, self.out_dim)
            for ntype, feat in node_feats.items()
        }
        rel_feats = {r: self.fc_rel[r[1]](feat).view(-1, self.num_heads, self.out_dim) for r, feat in rel_feats.items()}

        # Combine features for each node type
        out_feats = {}
        for ntype, node_feat in node_feats.items():
            rel_node_feats = [feat for rel, feat in rel_feats.items() if rel[2] == ntype]
            if not rel_node_feats:
                continue
            elif len(rel_node_feats) == 1:
                out_feats[ntype] = rel_node_feats[0].view(-1, self.num_heads * self.out_dim)
            else:
                rel_node_feats = torch.stack(rel_node_feats, dim=0)
                cat_feats = torch.cat((node_feat.repeat(rel_node_feats.shape[0], 1, 1, 1), rel_node_feats), dim=-1)
                attn_scores = self.leaky_relu((self.attn * cat_feats).sum(dim=-1, keepdim=True))
                attn_scores = F.softmax(attn_scores, dim=0)
                out_feat = (attn_scores * rel_node_feats).sum(dim=0)
                out_feats[ntype] = self.dropout(out_feat.reshape(-1, self.num_heads * self.out_dim))
        return out_feats


class HGConvEncoder(BaseEncoder):
    """HGConv encoder for heterogeneous graphs."""

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.2,
        residual: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.residual = residual

        # Input projection
        self.fc_in = nn.ModuleDict(
            {ntype: nn.Linear(in_dim, num_heads * hidden_dim) for ntype, in_dim in in_dims.items()}
        )

        # HGConv layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = self._build_conv_layer(num_heads * hidden_dim, hidden_dim, num_heads, dropout)
            self.layers.append(layer)

        self.reset_parameters()

    def _build_conv_layer(self, in_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        """Build a single HGConv layer."""
        # Create parameters for micro-level convolution
        micro_fc = {ntype: nn.Linear(in_dim, num_heads * hidden_dim, bias=False) for ntype in self.fc_in.keys()}
        micro_attn = {
            ntype: nn.Parameter(torch.FloatTensor(size=(num_heads, 2 * hidden_dim))) for ntype in self.fc_in.keys()
        }

        # Create parameters for macro-level convolution
        macro_fc_node = nn.ModuleDict(
            {ntype: nn.Linear(in_dim, num_heads * hidden_dim, bias=False) for ntype in self.fc_in.keys()}
        )
        macro_fc_rel = nn.ModuleDict()  # Will be populated during forward pass
        macro_attn = nn.Parameter(torch.FloatTensor(size=(num_heads, 2 * hidden_dim)))

        return {
            "micro_fc": micro_fc,
            "micro_attn": micro_attn,
            "macro_fc_node": macro_fc_node,
            "macro_fc_rel": macro_fc_rel,
            "macro_attn": macro_attn,
            "dropout": dropout,
        }

    def reset_parameters(self):
        """Initialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        for fc in self.fc_in.values():
            nn.init.xavier_normal_(fc.weight, gain=gain)
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)

    def forward(
        self,
        g: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        return_all_nodes: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward computation."""
        # Initial feature transformation
        h_dict = {ntype: self.fc_in[ntype](feat) for ntype, feat in feat_dict.items()}

        # Process each layer
        for layer in self.layers:
            h_new = {}
            # Micro-level convolution for each relation
            rel_feats = {}
            for stype, etype, dtype in g.canonical_etypes:
                if g.num_edges((stype, etype, dtype)) > 0:
                    micro_conv = MicroConv(
                        self.hidden_dim,
                        self.num_heads,
                        layer["micro_fc"][stype],
                        layer["micro_fc"][dtype],
                        layer["micro_attn"][stype],
                        layer["dropout"],
                        activation=F.relu,
                    )
                    rel_feats[(stype, etype, dtype)] = micro_conv(g[stype, etype, dtype], h_dict[stype], h_dict[dtype])

            # Macro-level convolution
            macro_conv = MacroConv(
                self.hidden_dim,
                self.num_heads,
                layer["macro_fc_node"],
                layer["macro_fc_rel"],
                layer["macro_attn"],
                layer["dropout"],
            )
            h_dict = macro_conv(h_dict, rel_feats)

        return h_dict if return_all_nodes else h_dict["paper"]


class HGConvRecommender(BaseRecommender):
    """HGConv-based paper recommender."""

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        temperature: float = 0.1,
        dropout: float = 0.2,
    ):
        encoder = HGConvEncoder(
            in_dims=in_dims,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        super().__init__(encoder)
        self.criterion = InfoNCELoss(temperature=temperature)

    def forward(
        self,
        g: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        batch_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward computation for both training and inference."""
        paper_embeds = self.encoder(g, feat_dict)

        if batch_data is None:
            return paper_embeds

        # Get batch embeddings for contrastive learning
        anchor_embeds = paper_embeds[batch_data["anchor"]]
        pos_embeds = paper_embeds[batch_data["positive"]]
        neg_embeds = paper_embeds[batch_data["negative"].view(-1)].view(
            len(anchor_embeds), -1, self.encoder.hidden_dim * self.encoder.num_heads
        )

        # Compute loss
        loss = self.criterion(anchor_embeds, pos_embeds, neg_embeds)
        return {"loss": loss}
