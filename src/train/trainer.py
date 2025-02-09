from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import json

import dgl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from src.config import settings
from src.models.base import BaseRecommender
from src.models.gnn.metrics import compute_metrics


class Trainer:
    """Trainer for paper recommendation models."""

    def __init__(
        self,
        model: BaseRecommender,
        device: torch.device,
        model_type: str,
        run_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize the trainer.

        Parameters
        ----------
        model : BaseRecommender
            The recommendation model to train
        device : torch.device
            Device to use for training
        model_type : str
            Type of the model
        run_dir : Path, optional
            Directory to save checkpoints and logs
        **kwargs : dict
            Additional keyword arguments
        """
        self.model = model
        self.device = device
        self.model_type = model_type
        
        # Setup run directory
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = settings.model_dir / model_type / f"run_{timestamp}"
        self.run_dir = run_dir
        self.checkpoint_dir = run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "model_type": model_type,
            "model_config": settings[model_type].dict(),
            "training_config": settings.training.dict(),
            **kwargs
        }
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        # Setup training
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=settings.training.learning_rate,
            weight_decay=settings.training.weight_decay
        )
        self.best_val_metric = float("-inf")
        self.metrics_history = []

    def train_epoch(
        self,
        g: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Parameters
        ----------
        g : dgl.DGLGraph
            The full graph
        feat_dict : Dict[str, torch.Tensor]
            Node features
        train_loader : DataLoader
            Training data loader

        Returns
        -------
        Dict[str, float]
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass and compute loss
            self.optimizer.zero_grad()
            outputs = self.model.train_step(g, feat_dict, batch)
            loss = outputs["loss"]

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
        }

    @torch.no_grad()
    def validate(
        self,
        g: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        val_pairs: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Run validation.

        Parameters
        ----------
        g : dgl.DGLGraph
            The full graph
        feat_dict : Dict[str, torch.Tensor]
            Node features
        val_pairs : torch.Tensor
            Validation paper pairs (N, 2)
        val_labels : torch.Tensor
            Ground truth labels for pairs

        Returns
        -------
        Dict[str, float]
            Dictionary containing validation metrics
        """
        self.model.eval()

        # Get embeddings
        paper_embeds = self.model.get_embeddings(g, feat_dict)

        # Compute similarities for validation pairs
        query_embeds = paper_embeds[val_pairs[:, 0]]
        paper_embeds = paper_embeds[val_pairs[:, 1]]
        sim = torch.mm(query_embeds, paper_embeds.t())

        # Compute metrics
        metrics = compute_metrics(sim, val_labels)
        return metrics

    def train(
        self,
        g: dgl.DGLGraph,
        feat_dict: Dict[str, torch.Tensor],
        train_loader: DataLoader,
        val_pairs: torch.Tensor,
        val_labels: torch.Tensor,
        num_epochs: int,
        patience: int = 10,
    ) -> None:
        """Train the model.

        Parameters
        ----------
        g : dgl.DGLGraph
            The full graph
        feat_dict : Dict[str, torch.Tensor]
            Node features
        train_loader : DataLoader
            Training data loader
        val_pairs : torch.Tensor
            Validation paper pairs
        val_labels : torch.Tensor
            Ground truth labels for validation pairs
        num_epochs : int
            Number of epochs to train
        patience : int
            Early stopping patience
        """
        # Move data to device
        g = g.to(self.device)
        feat_dict = {k: v.to(self.device) for k, v in feat_dict.items()}
        val_pairs = val_pairs.to(self.device)
        val_labels = val_labels.to(self.device)

        for epoch in range(num_epochs):
            # Train one epoch
            train_metrics = self.train_epoch(g, feat_dict, train_loader)

            # Validate
            val_metrics = self.validate(g, feat_dict, val_pairs, val_labels)
            val_metric = val_metrics["ndcg@10"]  # Use NDCG@10 for model selection

            # Log progress
            metrics_str = f"Epoch {epoch:03d}"
            metrics_str += f" | Loss {train_metrics['loss']:.4f}"
            metrics_str += f" | Val NDCG@10 {val_metric:.4f}"
            print(metrics_str)

            # Save best model and check early stopping
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.save_checkpoint("best.pt", val_metrics)
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print("Early stopping triggered")
                    break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt", val_metrics)

    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None) -> None:
        """Save model checkpoint and metrics."""
        # Save model
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_metric": self.best_val_metric,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

        # Save metrics
        if metrics is not None:
            self.metrics_history.append(metrics)
            with open(self.run_dir / "metrics.json", "w") as f:
                json.dump(self.metrics_history, f, indent=2)

    @classmethod
    def load_from_checkpoint(
        cls,
        run_dir: Path,
        model: BaseRecommender,
        device: torch.device,
    ) -> "Trainer":
        """Load trainer from a previous run."""
        # Load config
        with open(run_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        # Create trainer
        trainer = cls(model, device, config["model_type"], run_dir)

        # Load checkpoint
        checkpoint = torch.load(run_dir / "checkpoints/best.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.best_val_metric = checkpoint["best_val_metric"]

        return trainer
