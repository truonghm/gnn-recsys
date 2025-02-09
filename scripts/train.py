import torch
from torch.utils.data import DataLoader

from src.config import settings
from src.data.dataset import OAGCSDataset
from src.models.gnn.model import HGTRecommender
from src.train.sampler import ContrastiveSampler
from src.train.trainer import Trainer


def main():
    # 1. Load dataset
    dataset = OAGCSDataset()
    g = dataset.graph
    feat_dict = {ntype: g.nodes[ntype].data["feat"] for ntype in g.ntypes}

    # 2. Create model
    model = HGTRecommender(
        in_dims={ntype: feat_dict[ntype].shape[1] for ntype in g.ntypes},
        hidden_dim=settings.model.hidden_dim,
        num_layers=settings.model.num_layers,
        num_heads=settings.model.num_heads,
        dropout=settings.model.dropout,
    )

    # 3. Setup training
    device = torch.device(settings.device)

    # Create sampler for contrastive learning
    sampler = ContrastiveSampler(
        g=g,
        batch_size=settings.training.batch_size,
        num_negative=settings.training.num_negative,
        share_metadata=[("paper", "has_field", "field"), ("paper", "published_at", "venue")],
    )
    train_loader = DataLoader(
        range(g.num_nodes("paper")),
        batch_sampler=sampler,
    )

    # Create validation pairs
    # For simplicity, use citation links as validation pairs
    src, dst = g.edges(etype=("paper", "cites", "paper"))
    val_pairs = torch.stack([src, dst], dim=1)
    val_labels = torch.ones(len(src))

    # 4. Create trainer and train
    trainer = Trainer(
        model=model,
        device=device,
        lr=settings.training.lr,
        weight_decay=settings.training.weight_decay,
    )

    trainer.train(
        g=g,
        feat_dict=feat_dict,
        train_loader=train_loader,
        val_pairs=val_pairs,
        val_labels=val_labels,
        num_epochs=settings.training.num_epochs,
        patience=settings.training.patience,
    )


if __name__ == "__main__":
    main()
