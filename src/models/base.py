from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import dgl
import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Base encoder class for graph neural networks.

    This abstract class defines the interface that all GNN encoders must implement.
    Encoders are responsible for generating node embeddings from the input graph.
    """

    @abstractmethod
    def forward(
        self,
        g: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        return_all_nodes: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass to generate node embeddings.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input heterogeneous graph
        feat_dict : Dict[str, torch.Tensor]
            Node features for each node type
        return_all_nodes : bool, optional
            If True, return embeddings for all node types,
            otherwise return only paper embeddings

        Returns
        -------
        Union[torch.Tensor, Dict[str, torch.Tensor]]
            If return_all_nodes is False:
                Paper embeddings tensor (n_papers, hidden_dim)
            If return_all_nodes is True:
                Dictionary of embeddings for each node type
        """
        raise NotImplementedError


class BaseRecommender(nn.Module, ABC):
    """Base recommender class for paper recommendation models.

    This abstract class defines the interface that all recommendation models
    must implement. Recommenders combine an encoder with training and
    inference logic.
    """

    def __init__(self, encoder: BaseEncoder):
        """Initialize the recommender.

        Parameters
        ----------
        encoder : BaseEncoder
            The GNN encoder to use for generating embeddings
        """
        super().__init__()
        self.encoder = encoder

    @abstractmethod
    def forward(
        self,
        g: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        batch_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for both training and inference.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input heterogeneous graph
        feat_dict : Dict[str, torch.Tensor]
            Node features for each node type
        batch_data : Dict[str, torch.Tensor], optional
            Training batch data (e.g., anchor/pos/neg indices)
            If None, run in inference mode

        Returns
        -------
        Union[torch.Tensor, Dict[str, torch.Tensor]]
            During training:
                Dictionary with model-specific outputs (e.g., losses)
            During inference:
                Paper embeddings tensor (n_papers, hidden_dim)
        """
        raise NotImplementedError

    def get_embeddings(
        self,
        g: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        paper_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get paper embeddings for recommendation.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input heterogeneous graph
        feat_dict : Dict[str, torch.Tensor]
            Node features for each node type
        paper_idx : torch.Tensor, optional
            If provided, only return embeddings for these paper indices

        Returns
        -------
        torch.Tensor
            Paper embeddings (n_papers, hidden_dim) or (len(paper_idx), hidden_dim)
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.encoder(g, feat_dict)
            if paper_idx is not None:
                embeddings = embeddings[paper_idx]
        return embeddings

    def train_step(
        self,
        g: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        batch_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Perform a single training step.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input heterogeneous graph
        feat_dict : Dict[str, torch.Tensor]
            Node features for each node type
        batch_data : Dict[str, torch.Tensor]
            Training batch data

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing loss and any other metrics
        """
        self.train()
        return self.forward(g, feat_dict, batch_data)
