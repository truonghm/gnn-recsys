from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BaseModelConfig(BaseModel):
    """Base configuration shared by all models"""
    hidden_dim: int = Field(default=64, description="Hidden dimension size")
    num_layers: int = Field(default=2, description="Number of GNN layers")
    dropout: float = Field(default=0.2, description="Dropout rate")


class HGTConfig(BaseModelConfig):
    """HGT model configuration"""
    num_heads: int = Field(default=8, description="Number of attention heads")
    negative_slope: float = Field(default=0.2, description="LeakyReLU negative slope")


class HGConvConfig(BaseModelConfig):
    """HGConv model configuration"""
    num_heads: int = Field(default=8, description="Number of attention heads")
    residual: bool = Field(default=True, description="Use residual connections")


class HeCoConfig(BaseModelConfig):
    """HeCo model configuration"""
    num_metapaths: int = Field(default=4, description="Number of metapaths")
    lambda_: float = Field(default=0.5, description="Balance between two views")
    feat_drop: float = Field(default=0.3, description="Feature dropout")
    attn_drop: float = Field(default=0.5, description="Attention dropout")


class TrainingConfig(BaseModel):
    """Training configuration"""
    num_epochs: int = Field(default=100, description="Number of training epochs")
    batch_size: int = Field(default=1024, description="Batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    weight_decay: float = Field(default=0.0, description="Weight decay")
    num_negative_samples: int = Field(default=5, description="Number of negative samples per positive")
    temperature: float = Field(default=0.07, description="Temperature for InfoNCE loss")
    early_stopping_patience: int = Field(default=10, description="Early stopping patience")


class DataConfig(BaseModel):
    """Data configuration"""
    train_ratio: float = Field(default=0.8, description="Training set ratio")
    val_ratio: float = Field(default=0.1, description="Validation set ratio")
    test_ratio: float = Field(default=0.1, description="Test set ratio")
    min_citations: int = Field(default=1, description="Minimum citations for papers")
    max_year: int = Field(default=2021, description="Latest year to include")
    min_year: int = Field(default=2010, description="Earliest year to include")
    raw_data_dir: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_data_dir: Path = Field(default=Path("data/processed"), description="Processed data directory")
    cache_dir: Path = Field(default=Path("data/cache"), description="Cache directory")


class Settings(BaseModel):
    """Global settings"""
    # Paths
    base_dir: Path = Field(default=Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    model_dir: Path = Field(default=Path("models"), description="Model save directory")
    output_dir: Path = Field(default=Path("outputs"), description="Output directory")

    # Configurations
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    # Model configurations
    model_type: str = Field(default="hgt", description="Model architecture (hgt/hgconv/heco)")
    hgt: HGTConfig = Field(default_factory=HGTConfig)
    hgconv: HGConvConfig = Field(default_factory=HGConvConfig)
    heco: HeCoConfig = Field(default_factory=HeCoConfig)

    def update_from_yaml(self, yaml_path: Path) -> None:
        """Update settings from YAML file"""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def device(self) -> str:
        """Get device string based on CUDA availability"""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to attributes"""
        return getattr(self, key)


# Global settings instance
settings = Settings()
