# GNN-Based Academic Paper Recommendation System

A graph neural network-based system for recommending academic papers based on various academic entities (papers, authors, venues, etc.).

## Project Overview

This project aims to build a flexible academic paper recommendation system that can:

1. Take any academic entity (paper, author, venue, field, institution) as input
2. Generate recommendations for the most relevant research papers
3. Leverage the rich heterogeneous graph structure of academic data
4. Use state-of-the-art graph neural network architectures

## Model Architecture

The system uses a multi-stage architecture combining heterogeneous graph neural networks with contrastive learning:

1. **Graph Construction**
   - Heterogeneous graph with 5 types of nodes (papers, authors, venues, fields, institutions)
   - Multiple edge types capturing different academic relationships
   - Citation network backbone enhanced with metadata relationships

2. **Node Feature Generation**
   - Papers: Text embeddings from titles and abstracts 
   - Authors: Aggregated from authored papers
   - Venues: Learned embeddings
   - Fields: Learned embeddings
   - Institutions: Learned embeddings

3. **Graph Neural Network**
   - Relation-aware heterogeneous GNN layers
   - Multi-head attention for different relationship types
   - Skip connections and layer normalization
   - Outputs node embeddings that capture both structure and semantics

4. **Contrastive Learning**
   - Positive samples: Papers with high citation-based similarity
   - Negative samples: Random papers + hard negatives
   - InfoNCE loss for training embeddings

5. **Recommendation Generation**
   - Similarity computation between query entity and candidate papers
   - Re-ranking with citation counts and temporal factors
   - Top-K selection for final recommendations

## Data Usage

The dataset is split into training, validation and test sets:

### Training Set (80%)
- Used for training the GNN model
- Contains complete citation information
- All node types and relationships included

### Validation Set (10%) 
- Used for hyperparameter tuning
- Citation links to training set preserved
- Metrics: NDCG@K, Recall@K

### Test Set (10%)
- Used for final evaluation
- Citation links to training set preserved
- Completely held-out from training
- Metrics: NDCG@K, Recall@K, MRR

### Ground Truth
- Citation relationships serve as ground truth
- Papers that cite each other are considered relevant
- Citation count used as relevance weight
- Time-aware evaluation (emphasize recent papers)

## Data

The project uses the Microsoft Academic Graph (MAG) dataset, specifically filtered for Computer Science papers. The heterogeneous graph includes:

### Vertices
- Papers: Research papers with titles, abstracts, and other metadata
- Authors: Researchers who wrote the papers
- Venues: Journals and conferences where papers were published
- Fields: Research fields and topics
- Institutions: Academic and research institutions

### Edges
- Author-writes->Paper: Authorship relationships
- Paper-published_at->Venue: Publication venues
- Paper-has_field->Field: Research field classifications
- Paper-cites->Paper: Citation relationships
- Author-affiliated_with->Institution: Author affiliations

## Project Structure

```
 src/
 ├── config.py           # Global configuration and settings management
 │
 ├── data/               # Data Processing Layer
 │   ├── dataset.py      # DGL dataset implementation for OAG CS
 │   ├── transforms.py   # Graph and feature transformations
 │   └── oag_cs/         # Raw data processing for OAG CS dataset
 │
 ├── models/             # Model Layer
 │   ├── gnn/           
 │   │   ├── layers.py   # Basic GNN layer implementations
 │   │   └── model.py    # Full model architecture
 │   └── metrics.py      # Scoring functions and evaluation metrics (ranking metrics, losses)
 │
 ├── train/              # Training Layer
 │   ├── trainer.py      # Training loop and optimization
 │   └── sampler.py      # Contrastive learning samplers
 │
 └── recommend/          # Recommendation Layer
     ├── engine.py       # Recommendation generation logic
     └── ranking.py      # Ranking and re-ranking strategies
```

### Layer Descriptions

1. **Data Processing Layer**
   - Handles raw academic data processing and cleaning
   - Constructs heterogeneous graph structure
   - Manages feature generation and transformation
   - Implements data splitting and sampling strategies

2. **Model Layer**
   - Implements core GNN architecture components
   - Provides attention mechanisms for heterogeneous graphs
   - Defines loss functions and model interfaces
   - Includes evaluation metrics and validation logic

3. **Training Layer**
   - Manages training loops and optimization
   - Implements contrastive learning strategies
   - Handles batch sampling and negative mining
   - Provides checkpointing and model selection

4. **Recommendation Layer**
   - Generates paper recommendations
   - Implements similarity computation
   - Provides re-ranking strategies
   - Handles recommendation filtering and post-processing

### Key Components

- **Dataset (data/dataset.py)**
  - DGL dataset class for OAG CS data
  - Heterogeneous graph construction
  - Feature processing and caching

- **GNN Model (models/gnn/model.py)**
  - Relation-aware graph neural networks
  - Multi-head attention mechanisms
  - Contrastive learning objectives

- **Training Pipeline (train/trainer.py)**
  - Efficient mini-batch training
  - Positive/negative sampling
  - Model evaluation and selection

- **Recommendation Engine (recommend/engine.py)**
  - Entity-to-paper recommendation
  - Similarity-based ranking
  - Time-aware re-ranking

## Key Features

1. **Heterogeneous Graph Modeling**
   - Captures complex relationships between academic entities
   - Preserves rich semantic information in the academic network

2. **Flexible Entity-based Recommendations**
   - Can generate paper recommendations from any entity type
   - Supports multiple recommendation scenarios

3. **Advanced GNN Architecture**
   - Handles heterogeneous graph structure
   - Learns effective node embeddings
   - Incorporates attention mechanisms

4. **Evaluation Framework**
   - Standard ranking metrics (NDCG, Recall)
   - Domain-specific evaluation metrics
   - Comprehensive validation pipeline

## Baseline models

- HGConv
- HeCo
- Homogeneous GAT
- Content-based

## Getting Started

### Prerequisites
- Python 3.10
- PyTorch
- DGL (Deep Graph Library)
- Other dependencies listed in pyproject.toml and uv.lock
