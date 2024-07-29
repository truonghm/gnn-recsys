"""
RHGNN: Relation-aware Heterogeneous Graph Neural Network

This module implements the RHGNN algorithm for learning on heterogeneous graphs. Key components:

- RelationGraphConv: Performs convolution for specific relations
- RelationCrossing: Handles cross-relation message passing
- RelationFusing: Mixes representations under different relations
- RHGNNLayer: Main layer combining the above components
- RHGNN: Full model with multiple layers and classification


Features:

- Relation-aware processing
- Multi-head attention
- Residual connections
- Support for heterogeneous graphs
- Based on the paper: "Relation-aware Heterogeneous Graph Neural Network" (https://arxiv.org/pdf/2105.11122)

Uses DGL for graph operations and PyTorch for neural network computations.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.heterograph import DGLBlock
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair


class RelationGraphConv(nn.Module):
    def __init__(
        self,
        out_dim: int,
        num_heads: int,
        fc_src: nn.Linear,
        fc_dst: nn.Linear,
        fc_rel: nn.Linear,
        feat_drop: float = 0.0,
        negative_slope: float = 0.2,
        activation: Optional[Callable] = None,
    ) -> None:
        """
        Convolution for specific relations

        For a specific relation (edge type) R=<stype, etype, dtype>, aggregate neighbor information under relation R
        to obtain the representation of dtype type vertices under relation R.
        The attention vector uses the representation of relation R.

        Parameters
        ----------
        out_dim : int
            Output feature dimension.
        num_heads : int
            Number of attention heads K.
        fc_src : nn.Linear(d_in, K*d_out)
            Feature transformation module for source vertices.
        fc_dst : nn.Linear(d_in, K*d_out)
            Feature transformation module for destination vertices.
        fc_rel : nn.Linear(d_rel, 2*K*d_out)
            Transformation module for relation representation.
        feat_drop : float, optional
            Dropout probability for input features (default: 0).
        negative_slope : float, optional
            Negative slope for LeakyReLU (default: 0.2).
        activation : callable, optional
            Activation function for output features (default: None).
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.fc_src = fc_src
        self.fc_dst = fc_dst
        self.fc_rel = fc_rel
        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation

    def forward(
        self, g: DGLGraph, feat: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], feat_rel: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        g : DGLGraph
            Bipartite graph (containing only one type of relation)
        feat : tensor(N_src, d_in) or (tensor(N_src, d_in), tensor(N_dst, d_in))
            Input features
        feat_rel : tensor(d_rel)
            Representation of relation R

        Returns
        -------
        tensor(N_dst, K*d_out)
            Representation of target nodes under relation R
        """
        with g.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, g)
            feat_src = self.fc_src(self.feat_drop(feat_src)).view(-1, self.num_heads, self.out_dim)
            feat_dst = self.fc_dst(self.feat_drop(feat_dst)).view(-1, self.num_heads, self.out_dim)
            attn = self.fc_rel(feat_rel).view(self.num_heads, 2 * self.out_dim)

            # a^T (z_u || z_v) = (a_l^T || a_r^T) (z_u || z_v) = a_l^T z_u + a_r^T z_v = el + er
            el = (feat_src * attn[:, : self.out_dim]).sum(dim=-1, keepdim=True)  # (N_src, K, 1)
            er = (feat_dst * attn[:, self.out_dim :]).sum(dim=-1, keepdim=True)  # (N_dst, K, 1)
            g.srcdata.update({"ft": feat_src, "el": el})
            g.dstdata["er"] = er
            g.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(g.edata.pop("e"))
            g.edata["a"] = edge_softmax(g, e)  # (E, K, 1)

            # 消息传递
            g.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            ret = g.dstdata["ft"].view(-1, self.num_heads * self.out_dim)
            if self.activation:
                ret = self.activation(ret)
            return ret


class RelationCrossing(nn.Module):
    def __init__(
        self, out_dim: int, num_heads: int, rel_attn: nn.Parameter, dropout: float = 0.0, negative_slope: float = 0.2
    ) -> None:
        """
        Cross-relation message passing

        For a relation R=<stype, etype, dtype>, combine the representations of dtype vertices under different relations

        Parameters
        ----------
        out_dim : int
            Output feature dimension
        num_heads : int
            Number of attention heads K
        rel_attn : nn.Parameter(K, d)
            Attention vector for relation R
        dropout : float, optional
            Dropout probability, default is 0
        negative_slope : float, optional
            Negative slope for LeakyReLU, default is 0.2
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.rel_attn = rel_attn
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        feats : tensor
            Shape (N_R, N, K*d). Representations of dtype vertices under different relations.

        Returns
        -------
        tensor
            Shape (N, K*d). Representations of dtype vertices under relation R after cross-relation message passing.
        """
        num_rel = feats.shape[0]
        if num_rel == 1:
            return feats.squeeze(dim=0)
        feats = feats.view(num_rel, -1, self.num_heads, self.out_dim)  # (N_R, N, K, d)
        attn_scores = (self.rel_attn * feats).sum(dim=-1, keepdim=True)
        attn_scores = F.softmax(self.leaky_relu(attn_scores), dim=0)  # (N_R, N, K, 1)
        out = (attn_scores * feats).sum(dim=0)  # (N, K, d)
        out = self.dropout(out.view(-1, self.num_heads * self.out_dim))  # (N, K*d)
        return out


class RelationFusing(nn.Module):
    def __init__(
        self,
        node_hidden_dim: int,
        rel_hidden_dim: int,
        num_heads: int,
        w_node: Dict[str, torch.Tensor],
        w_rel: Dict[str, torch.Tensor],
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ) -> None:
        """
        Relation Mixing

        Combine representations of vertices of a specific type under different relations.

        Parameters
        ----------
        node_hidden_dim : int
            Dimension of node hidden features.
        rel_hidden_dim : int
            Dimension of relation hidden features.
        num_heads : int
            Number of attention heads K.
        w_node : Dict[str, tensor(K, d_node, d_node)]
            Mapping from edge types to feature transformation matrices for nodes in that relation.
        w_rel : Dict[str, tensor(K, d_rel, d_node)]
            Mapping from edge types to feature transformation matrices for relations.
        dropout : float, optional
            Dropout probability, default is 0.
        negative_slope : float, optional
            Negative slope for LeakyReLU, default is 0.2
        """
        super().__init__()
        self.node_hidden_dim = node_hidden_dim
        self.rel_hidden_dim = rel_hidden_dim
        self.num_heads = num_heads
        self.w_node = nn.ParameterDict(w_node)
        self.w_rel = nn.ParameterDict(w_rel)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, node_feats: Dict[str, torch.Tensor], rel_feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        node_feats : Dict[str, tensor(N, K*d_node)]
            Mapping from edge types to vertex representations under that relationship.
        rel_feats : Dict[str, tensor(K*d_rel)]
            Mapping from edge types to relationship representations.

        Returns
        -------
        tensor(N, K*d_node)
            Final embedding of vertices of this type.
        """
        etypes = list(node_feats.keys())
        num_rel = len(node_feats)
        if num_rel == 1:
            return node_feats[etypes[0]]
        node_feats = torch.stack([node_feats[e] for e in etypes], dim=0).reshape(
            num_rel, -1, self.num_heads, self.node_hidden_dim
        )  # (N_R, N, K, d_node)
        rel_feats = torch.stack([rel_feats[e] for e in etypes], dim=0).reshape(
            num_rel, self.num_heads, self.rel_hidden_dim
        )  # (N_R, K, d_rel)
        w_node = torch.stack([self.w_node[e] for e in etypes], dim=0)  # (N_R, K, d_node, d_node)
        w_rel = torch.stack([self.w_rel[e] for e in etypes], dim=0)  # (N_R, K, d_rel, d_node)

        # hn[r, n, h] @= wn[r, h] => hn[r, n, h, i] = ∑(k) hn[r, n, h, k] * wn[r, h, k, i]
        node_feats = torch.einsum("rnhk,rhki->rnhi", node_feats, w_node)  # (N_R, N, K, d_node)
        # hr[r, h] @= wr[r, h] => hr[r, h, i] = ∑(k) hr[r, h, k] * wr[r, h, k, i]
        rel_feats = torch.einsum("rhk,rhki->rhi", rel_feats, w_rel)  # (N_R, K, d_node)

        attn_scores = (node_feats * rel_feats.unsqueeze(dim=1)).sum(dim=-1, keepdim=True)
        attn_scores = F.softmax(self.leaky_relu(attn_scores), dim=0)  # (N_R, N, K, 1)
        out = (attn_scores * node_feats).sum(dim=0)  # (N_R, N, K, d_node)
        out = self.dropout(out.view(-1, self.num_heads * self.node_hidden_dim))  # (N, K*d_node)
        return out


class RHGNNLayer(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        node_out_dim: int,
        rel_in_dim: int,
        rel_out_dim: int,
        num_heads: int,
        ntypes: List[str],
        etypes: List[Tuple[str, str, str]],
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        residual: bool = True,
    ) -> None:
        """
        R-HGNN layer

        Parameters
        ----------
        node_in_dim : int
            Input dimension of node features
        node_out_dim : int
            Output dimension of node features
        rel_in_dim : int
            Input dimension of relation features
        rel_out_dim : int
            Output dimension of relation features
        num_heads : int
            Number of attention heads K
        ntypes : List[str]
            List of node types
        etypes : List[(str, str, str)]
            List of canonical edge types
        dropout : float, optional
            Dropout probability, default is 0
        negative_slope : float, optional
            Negative slope of LeakyReLU, default is 0.2
        residual : bool, optional
            Whether to use residual connection, default is True
        """
        super().__init__()
        # Parameters for convolution of specific relations
        fc_node = {ntype: nn.Linear(node_in_dim, num_heads * node_out_dim, bias=False) for ntype in ntypes}
        fc_rel = {etype: nn.Linear(rel_in_dim, 2 * num_heads * node_out_dim, bias=False) for _, etype, _ in etypes}
        self.rel_graph_conv = nn.ModuleDict(
            {
                etype: RelationGraphConv(
                    node_out_dim,
                    num_heads,
                    fc_node[stype],
                    fc_node[dtype],
                    fc_rel[etype],
                    dropout,
                    negative_slope,
                    F.relu,
                )
                for stype, etype, dtype in etypes
            }
        )

        # Residual connection parameters
        self.residual = residual
        if residual:
            self.fc_res = nn.ModuleDict({ntype: nn.Linear(node_in_dim, num_heads * node_out_dim) for ntype in ntypes})
            self.res_weight = nn.ParameterDict({ntype: nn.Parameter(torch.rand(1)) for ntype in ntypes})

        # Parameters for relation representation learning
        self.fc_upd = nn.ModuleDict({etype: nn.Linear(rel_in_dim, num_heads * rel_out_dim) for _, etype, _ in etypes})

        # Parameters for inter-relation message passing
        rel_attn = {etype: nn.Parameter(torch.FloatTensor(num_heads, node_out_dim)) for _, etype, _ in etypes}
        self.rel_cross = nn.ModuleDict(
            {
                etype: RelationCrossing(node_out_dim, num_heads, rel_attn[etype], dropout, negative_slope)
                for _, etype, _ in etypes
            }
        )

        self.rev_etype = {e: next(re for rs, re, rd in etypes if rs == d and rd == s and re != e) for s, e, d in etypes}
        self.reset_parameters(rel_attn)

    def reset_parameters(self, rel_attn: Dict[str, nn.Parameter]) -> None:
        gain = nn.init.calculate_gain("relu")
        for etype in rel_attn:
            nn.init.xavier_normal_(rel_attn[etype], gain=gain)

    def forward(
        self, g: DGLGraph, feats: Dict[Tuple[str, str, str], torch.Tensor], rel_feats: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[Tuple[str, str, str], torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        g : DGLGraph
            Heterogeneous graph.
        feats : Dict[(str, str, str), tensor(N_i, d_in)]
            Mapping of relations (triples) to input features of target vertices.
        rel_feats : Dict[str, tensor(d_in_rel)]
            Mapping of edge types to input relation features.

        Returns
        -------
        Dict[(str, str, str), tensor(N_i, K*d_out)]
            Mapping of relations (triples) to representations of target vertices under that relation.
        Dict[str, tensor(K*d_out_rel)]
            Mapping of edge types to relation representations.
        """
        if g.is_block:
            feats_dst = {r: feats[r][: g.num_dst_nodes(r[2])] for r in feats}
        else:
            feats_dst = feats

        node_rel_feats = {
            (stype, etype, dtype): self.rel_graph_conv[etype](
                g[stype, etype, dtype],
                (feats[(dtype, self.rev_etype[etype], stype)], feats_dst[(stype, etype, dtype)]),
                rel_feats[etype],
            )
            for stype, etype, dtype in g.canonical_etypes
            if g.num_edges((stype, etype, dtype)) > 0
        }  # {rel: tensor(N_dst, K*d_out)}

        if self.residual:
            for stype, etype, dtype in node_rel_feats:
                alpha = torch.sigmoid(self.res_weight[dtype])
                inherit_feat = self.fc_res[dtype](feats_dst[(stype, etype, dtype)])
                node_rel_feats[(stype, etype, dtype)] = (
                    alpha * node_rel_feats[(stype, etype, dtype)] + (1 - alpha) * inherit_feat
                )

        out_feats = {}  # {rel: tensor(N_dst, K*d_out)}
        for stype, etype, dtype in node_rel_feats:
            dst_node_rel_feats = torch.stack(
                [node_rel_feats[r] for r in node_rel_feats if r[2] == dtype], dim=0
            )  # (N_Ri, N_i, K*d_out)
            out_feats[(stype, etype, dtype)] = self.rel_cross[etype](dst_node_rel_feats)

        rel_feats = {etype: self.fc_upd[etype](rel_feats[etype]) for etype in rel_feats}
        return out_feats, rel_feats


class RHGNN(nn.Module):
    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int,
        out_dim: int,
        rel_in_dim: int,
        rel_hidden_dim: int,
        num_heads: int,
        ntypes: List[str],
        etypes: List[Tuple[str, str, str]],
        predict_ntype: str,
        num_layers: int,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        residual: bool = True,
    ) -> None:
        """
        R-HGNN model

        Parameters
        ----------
        in_dims : Dict[str, int]
            Mapping of vertex types to input feature dimensions.
        hidden_dim : int
            Hidden feature dimension for vertices.
        out_dim : int
            Output feature dimension for vertices.
        rel_in_dim : int
            Input feature dimension for relations.
        rel_hidden_dim : int
            Hidden feature dimension for relations.
        num_heads : int
            Number of attention heads K.
        ntypes : List[str]
            List of vertex types.
        etypes : List[(str, str, str)]
            List of canonical edge types.
        predict_ntype : str
            Vertex type to be predicted.
        num_layers : int
            Number of layers.
        dropout : float, optional
            Dropout probability, default is 0.
        negative_slope : float, optional
            Negative slope for LeakyReLU, default is 0.2.
        residual : bool, optional
            Whether to use residual connections, default is True.
        """
        super().__init__()
        self._d = num_heads * hidden_dim
        self.etypes = etypes
        self.predict_ntype = predict_ntype
        # Align input feature dimensions
        self.fc_in = nn.ModuleDict(
            {ntype: nn.Linear(in_dim, num_heads * hidden_dim) for ntype, in_dim in in_dims.items()}
        )
        # Relation input features
        self.rel_embed = nn.ParameterDict(
            {etype: nn.Parameter(torch.FloatTensor(1, rel_in_dim)) for _, etype, _ in etypes}
        )

        self.layers = nn.ModuleList()
        self.layers.append(
            RHGNNLayer(
                num_heads * hidden_dim,
                hidden_dim,
                rel_in_dim,
                rel_hidden_dim,
                num_heads,
                ntypes,
                etypes,
                dropout,
                negative_slope,
                residual,
            )
        )
        for _ in range(1, num_layers):
            self.layers.append(
                RHGNNLayer(
                    num_heads * hidden_dim,
                    hidden_dim,
                    num_heads * rel_hidden_dim,
                    rel_hidden_dim,
                    num_heads,
                    ntypes,
                    etypes,
                    dropout,
                    negative_slope,
                    residual,
                )
            )

        w_node = {etype: nn.Parameter(torch.FloatTensor(num_heads, hidden_dim, hidden_dim)) for _, etype, _ in etypes}
        w_rel = {
            etype: nn.Parameter(torch.FloatTensor(num_heads, rel_hidden_dim, hidden_dim)) for _, etype, _ in etypes
        }
        self.rel_fusing = nn.ModuleDict(
            {
                ntype: RelationFusing(
                    hidden_dim,
                    rel_hidden_dim,
                    num_heads,
                    {e: w_node[e] for _, e, d in etypes if d == ntype},
                    {e: w_rel[e] for _, e, d in etypes if d == ntype},
                    dropout,
                    negative_slope,
                )
                for ntype in ntypes
            }
        )
        self.classifier = nn.Linear(num_heads * hidden_dim, out_dim)
        self.reset_parameters(self.rel_embed, w_node, w_rel)

    def reset_parameters(
        self, rel_embed: nn.ParameterDict, w_node: Dict[str, nn.Parameter], w_rel: Dict[str, nn.Parameter]
    ) -> None:
        gain = nn.init.calculate_gain("relu")
        for etype in rel_embed:
            nn.init.xavier_normal_(rel_embed[etype], gain=gain)
            nn.init.xavier_normal_(w_node[etype], gain=gain)
            nn.init.xavier_normal_(w_rel[etype], gain=gain)

    def forward(
        self, blocks: List[DGLBlock], feats: Dict[str, torch.Tensor], return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        blocks : List[DGLBlock]
            List of DGL blocks.
        feats : Dict[str, tensor(N_i, d_in_i)]
            Mapping of node types to input node features.
        return_dict : bool, optional
            If True, returns a mapping of node types to final embeddings.
            If False, returns only the final embeddings of the node type to be predicted.

        Returns
        -------
        tensor(N_i, d_out) or Dict[str, tensor(N_i, d_out)]
            Final node embeddings.
        """
        feats = {(stype, etype, dtype): self.fc_in[dtype](feats[dtype]) for stype, etype, dtype in self.etypes}
        rel_feats = {rel: emb.flatten() for rel, emb in self.rel_embed.items()}
        for block, layer in zip(blocks, self.layers):
            # {(stype, etype, dtype): tensor(N_i, K*d_hid)}, {etype: tensor(K*d_hid_rel)}
            feats, rel_feats = layer(block, feats, rel_feats)

        out_feats = {
            ntype: self.rel_fusing[ntype](
                {e: feats[(s, e, d)] for s, e, d in feats if d == ntype},
                {e: rel_feats[e] for s, e, d in feats if d == ntype},
            )
            for ntype in set(d for _, _, d in feats)
        }  # {ntype: tensor(N_i, K*d_hid)}
        if return_dict:
            return {ntype: self.classifier(out_feat) for ntype, out_feat in out_feats.items()}
        else:
            return self.classifier(out_feats[self.predict_ntype])


class RHGNNFull(RHGNN):
    def forward(
        self, g: DGLGraph, feats: Dict[str, torch.Tensor], return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        return super().forward([g] * len(self.layers), feats, return_dict)
