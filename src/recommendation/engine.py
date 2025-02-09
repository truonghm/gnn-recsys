from typing import Any, Dict, List, Optional, Union

import dgl
import torch
import torch.nn as nn

from src.recommendation.ranking import diversity_rerank, rank_papers


class RecommendationEngine:
    """Engine for generating paper recommendations."""

    def __init__(
        self,
        model: nn.Module,
        graph: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        paper_years: Optional[torch.Tensor] = None,
        paper_citations: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize the recommendation engine.

        Parameters
        ----------
        model : nn.Module
            Trained GNN model
        graph : dgl.DGLGraph
            Full heterogeneous graph
        feat_dict : Dict[str, torch.Tensor]
            Node features
        paper_years : torch.Tensor, optional
            Publication years for papers
        paper_citations : torch.Tensor, optional
            Citation counts for papers
        device : torch.device, optional
            Device to use for computation
        """
        self.model = model
        self.graph = graph
        self.feat_dict = feat_dict
        self.paper_years = paper_years
        self.paper_citations = paper_citations
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model and data to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Pre-compute paper embeddings
        with torch.no_grad():
            self.paper_embeds = self.model(self.graph, self.feat_dict)

    def recommend_papers(
        self,
        query_papers: Union[torch.Tensor, List[int]],
        k: int = 10,
        time_weight: float = 0.1,
        citation_weight: float = 0.1,
        diversity_weight: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate paper recommendations.

        Parameters
        ----------
        query_papers : Union[torch.Tensor, List[int]]
            Paper indices to use as queries
        k : int
            Number of recommendations to return
        time_weight : float
            Weight for time-based reranking
        citation_weight : float
            Weight for citation-based reranking
        diversity_weight : float, optional
            Weight for diversity reranking (None = no diversity reranking)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - indices: Recommended paper indices (n_queries, k)
            - scores: Recommendation scores (n_queries, k)
        """
        if isinstance(query_papers, list):
            query_papers = torch.tensor(query_papers, device=self.device)

        # Get query embeddings
        query_embeds = self.paper_embeds[query_papers]

        # Initial ranking
        indices, scores = rank_papers(
            query_embeds,
            self.paper_embeds,
            self.paper_years,
            self.paper_citations,
            time_weight=time_weight,
            citation_weight=citation_weight,
            k=k if diversity_weight is None else k * 2,  # Get more candidates if using diversity reranking
        )

        # Diversity reranking
        if diversity_weight is not None:
            indices = diversity_rerank(query_embeds, self.paper_embeds, indices, lambda_div=1 - diversity_weight, k=k)
            # Recompute scores for final ranking
            scores = torch.gather(torch.mm(query_embeds, self.paper_embeds.t()), 1, indices)

        return {"indices": indices, "scores": scores}

    def batch_recommend(
        self, query_papers: Union[torch.Tensor, List[int]], batch_size: int = 1024, **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """Generate recommendations in batches.

        Parameters are same as recommend_papers() except:

        Parameters
        ----------
        batch_size : int
            Batch size for processing
        """
        if isinstance(query_papers, list):
            query_papers = torch.tensor(query_papers, device=self.device)

        all_indices = []
        all_scores = []

        for i in range(0, len(query_papers), batch_size):
            batch_queries = query_papers[i : i + batch_size]
            results = self.recommend_papers(batch_queries, **kwargs)
            all_indices.append(results["indices"])
            all_scores.append(results["scores"])

        return {"indices": torch.cat(all_indices, dim=0), "scores": torch.cat(all_scores, dim=0)}
