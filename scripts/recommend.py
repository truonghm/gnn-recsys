import torch

from src.config import settings
from src.data.dataset import OAGCSDataset
from src.models.gnn.model import HGTRecommender
from src.recommendation.engine import RecommendationEngine


def main():
    # 1. Load dataset and model
    dataset = OAGCSDataset()
    g = dataset.graph
    feat_dict = {ntype: g.nodes[ntype].data["feat"] for ntype in g.ntypes}

    # Create model with same config as training
    model = HGTRecommender(
        in_dims={ntype: feat_dict[ntype].shape[1] for ntype in g.ntypes},
        hidden_dim=settings.model.hidden_dim,
        num_layers=settings.model.num_layers,
        num_heads=settings.model.num_heads,
    )

    # Load trained weights
    checkpoint = torch.load(settings.model_dir / "best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # 2. Create recommendation engine
    engine = RecommendationEngine(
        model=model,
        graph=g,
        feat_dict=feat_dict,
        paper_years=g.nodes["paper"].data.get("year"),
        paper_citations=g.nodes["paper"].data.get("citation_count"),
    )

    # 3. Generate recommendations
    # Example: recommend papers similar to papers with IDs [0, 1, 2]
    query_papers = [0, 1, 2]
    recommendations = engine.recommend_papers(
        query_papers=query_papers, k=10, time_weight=0.1, citation_weight=0.1, diversity_weight=0.2
    )

    # Print results
    for i, query_id in enumerate(query_papers):
        print(f"\nRecommendations for paper {query_id}:")
        indices = recommendations["indices"][i]
        scores = recommendations["scores"][i]
        for idx, score in zip(indices, scores):
            print(f"Paper {idx.item()}: score = {score.item():.3f}")


if __name__ == "__main__":
    main()
