from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def rank_papers(
    query_embeds: torch.Tensor,
    paper_embeds: torch.Tensor,
    paper_years: Optional[torch.Tensor] = None,
    paper_citations: Optional[torch.Tensor] = None,
    time_weight: float = 0.1,
    citation_weight: float = 0.1,
    k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rank papers based on embedding similarity and optional metadata.

    Parameters
    ----------
    query_embeds : torch.Tensor
        Query embeddings (n_queries, hidden_dim)
    paper_embeds : torch.Tensor
        Paper embeddings (n_papers, hidden_dim)
    paper_years : torch.Tensor, optional
        Publication years (n_papers,)
    paper_citations : torch.Tensor, optional
        Citation counts (n_papers,)
    time_weight : float
        Weight for time-based reranking
    citation_weight : float
        Weight for citation-based reranking
    k : int
        Number of papers to return per query

    Returns
    -------
    torch.Tensor
        Indices of top-k papers (n_queries, k)
    torch.Tensor
        Scores of top-k papers (n_queries, k)
    """
    # Normalize embeddings
    query_embeds = F.normalize(query_embeds, p=2, dim=1)
    paper_embeds = F.normalize(paper_embeds, p=2, dim=1)

    # Compute similarity scores
    sim_scores = torch.mm(query_embeds, paper_embeds.t())  # (n_queries, n_papers)

    # Apply time-based reranking if years are provided
    if paper_years is not None:
        current_year = paper_years.max()
        time_decay = torch.exp(-(current_year - paper_years) * time_weight)
        sim_scores = sim_scores * time_decay.unsqueeze(0)

    # Apply citation-based reranking if citations are provided
    if paper_citations is not None:
        # Log-scale citation counts and normalize
        citation_scores = torch.log1p(paper_citations.float())
        citation_scores = citation_scores / citation_scores.max()
        sim_scores = sim_scores * (1 + citation_weight * citation_scores.unsqueeze(0))

    # Get top-k papers
    scores, indices = torch.topk(sim_scores, k=k, dim=1)
    return indices, scores


def diversity_rerank(
    query_embeds: torch.Tensor,
    paper_embeds: torch.Tensor,
    initial_indices: torch.Tensor,
    lambda_div: float = 0.5,
    k: Optional[int] = None,
) -> torch.Tensor:
    """Rerank papers to promote diversity using MMR.

    Implements Maximal Marginal Relevance (MMR) reranking to balance
    relevance and diversity.

    Parameters
    ----------
    query_embeds : torch.Tensor
        Query embeddings (n_queries, hidden_dim)
    paper_embeds : torch.Tensor
        Paper embeddings (n_papers, hidden_dim)
    initial_indices : torch.Tensor
        Initial ranking indices (n_queries, initial_k)
    lambda_div : float
        Diversity weight (0 = max diversity, 1 = max relevance)
    k : int, optional
        Number of papers to return, defaults to initial_k

    Returns
    -------
    torch.Tensor
        Reranked indices (n_queries, k)
    """
    if k is None:
        k = initial_indices.size(1)

    n_queries = query_embeds.size(0)
    device = query_embeds.device

    # Normalize embeddings
    query_embeds = F.normalize(query_embeds, p=2, dim=1)
    paper_embeds = F.normalize(paper_embeds, p=2, dim=1)

    final_indices = torch.zeros((n_queries, k), dtype=torch.long, device=device)

    for i in range(n_queries):
        query_embed = query_embeds[i]
        candidate_indices = initial_indices[i]
        candidate_embeds = paper_embeds[candidate_indices]

        # Compute relevance scores
        relevance = torch.mm(candidate_embeds, query_embed.unsqueeze(1)).squeeze()

        # Initialize selected set
        selected: List[int] = []
        remaining = list(range(len(candidate_indices)))

        for _ in range(k):
            if not remaining:
                break

            # Compute diversity penalty
            if selected:
                selected_embeds = paper_embeds[candidate_indices[selected]]
                similarity_to_selected = torch.max(torch.mm(candidate_embeds[remaining], selected_embeds.t()), dim=1)[0]
            else:
                similarity_to_selected = torch.zeros(len(remaining), device=device)

            # Compute MMR scores
            mmr_scores = lambda_div * relevance[remaining] - (1 - lambda_div) * similarity_to_selected

            # Select the best scoring paper
            best_idx = remaining[mmr_scores.argmax()]
            selected.append(best_idx)
            remaining.remove(best_idx)

        final_indices[i] = candidate_indices[torch.tensor(selected, device=device)]

    return final_indices
