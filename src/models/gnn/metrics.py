from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityScorer(nn.Module):
    """Score papers using cosine similarity with a temperature parameter."""

    def __init__(self, temperature: float = 0.1):
        """Initialize the scorer.

        Parameters
        ----------
        temperature : float
            Temperature parameter for scaling similarity scores
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embeds: torch.Tensor, paper_embeds: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores between queries and papers.

        Parameters
        ----------
        query_embeds : torch.Tensor
            Query embeddings (N_queries, hidden_dim)
        paper_embeds : torch.Tensor
            Paper embeddings (N_papers, hidden_dim)

        Returns
        -------
        torch.Tensor
            Similarity scores (N_queries, N_papers)
        """
        # Normalize embeddings
        query_embeds = F.normalize(query_embeds, p=2, dim=1)
        paper_embeds = F.normalize(paper_embeds, p=2, dim=1)

        # Compute cosine similarity
        sim = torch.mm(query_embeds, paper_embeds.t()) / self.temperature
        return sim


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""

    def __init__(self, temperature: float = 0.1):
        """Initialize the loss.

        Parameters
        ----------
        temperature : float
            Temperature parameter for scaling similarity scores
        """
        super().__init__()
        self.temperature = temperature
        self.scorer = CosineSimilarityScorer(temperature)

    def forward(
        self,
        query_embeds: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Parameters
        ----------
        query_embeds : torch.Tensor
            Query embeddings (N, hidden_dim)
        pos_embeds : torch.Tensor
            Positive example embeddings (N, hidden_dim)
        neg_embeds : torch.Tensor
            Negative example embeddings (N_neg, hidden_dim)

        Returns
        -------
        torch.Tensor
            InfoNCE loss value
        """
        # Compute similarities
        pos_sim = self.scorer(query_embeds, pos_embeds)  # (N, N)
        neg_sim = self.scorer(query_embeds, neg_embeds)  # (N, N_neg)

        # Gather positive similarities along diagonal
        pos_sim = torch.diag(pos_sim)  # (N,)

        # Compute loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (N, 1+N_neg)
        labels = torch.zeros(len(query_embeds), device=query_embeds.device, dtype=torch.long)

        return F.cross_entropy(logits, labels)


def compute_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """Compute ranking metrics for paper recommendations.

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted similarity scores (N_queries, N_papers)
    targets : torch.Tensor
        Ground truth relevance (N_queries, N_papers)
    k_values : List[int]
        Values of k for computing Precision@k, Recall@k, NDCG@k

    Returns
    -------
    Dict[str, float]
        Dictionary containing computed metrics
    """
    metrics = {}

    # Sort predictions
    _, indices = torch.sort(predictions, dim=1, descending=True)

    for k in k_values:
        # Get top-k predictions
        top_k = indices[:, :k]

        # Precision@k
        precision = torch.mean((torch.gather(targets, 1, top_k) > 0).float().mean(dim=1))
        metrics[f"precision@{k}"] = precision.item()

        # Recall@k
        recall = torch.mean(torch.gather(targets, 1, top_k).sum(dim=1) / (targets > 0).sum(dim=1).clamp(min=1))
        metrics[f"recall@{k}"] = recall.item()

        # NDCG@k
        dcg = torch.gather(targets, 1, top_k) / torch.log2(torch.arange(k, device=targets.device) + 2)
        dcg = dcg.sum(dim=1)

        # Compute ideal DCG
        ideal_dcg, _ = torch.sort(targets, dim=1, descending=True)
        ideal_dcg = ideal_dcg[:, :k] / torch.log2(torch.arange(k, device=targets.device) + 2)
        ideal_dcg = ideal_dcg.sum(dim=1).clamp(min=1e-6)

        ndcg = (dcg / ideal_dcg).mean()
        metrics[f"ndcg@{k}"] = ndcg.item()

    return metrics
