# Project Roadmap

## Overall Plan

* [ ] 2024/03: Collect resoureces (papers, datasets) and study foundational materials
* [ ] 2024/03: Explore and visualize datasets, crawl data, filter data for computer science topic only
* [ ] 2024/04: Start reading related papers and define clearly the structure of the project (preprocessing, training, evaluation, etc)
* [ ] 2024/04: Start implementing existing GNN models and recommendation algorithms

## Paper Reading

### Heterogeneous Graph Representation Learning

#### Surveys

* [ ] 2020 [Heterogeneous Network Representation Learning: A Unified Framework with Survey and Benchmark](https://arxiv.org/pdf/2004.00216)
* [ ] 2020 [A Survey on Heterogeneous Graph Embedding: Methods, Techniques, Applications and Sources](https://arxiv.org/pdf/2011.14867)

#### Graph Neural Networks

* [ ] 2014 [DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652)
* [ ] 2016 [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653)
* [ ] 2017 [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)
* [ ] 2017 [GCN: SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1609.02907)
* [ ] 2018 [R-GCN: Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/pdf/1703.06103)
* [ ] 2018 [GAT: GRAPH ATTENTION NETWORKS](https://arxiv.org/pdf/1710.10903)
* [ ] 2019 [HetGNN: Heterogeneous Graph Neural Network](https://dl.acm.org/doi/pdf/10.1145/3292500.3330961)
* [ ] 2019 [HAN: Heterogeneous Graph Attention Network](https://arxiv.org/pdf/1903.07293)
* [ ] 2020 [MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding](https://arxiv.org/pdf/2002.01680)
* [ ] 2020 [HGT: Heterogeneous Graph Transformer](https://arxiv.org/pdf/2003.01332)
* [ ] 2020 [HGConv: Hybrid Micro/Macro Level Convolution for Heterogeneous Graph Learning](https://arxiv.org/pdf/2012.14722)
* [ ] 2020 [GPT-GNN: Generative Pre-Training of Graph Neural Networks](https://arxiv.org/pdf/2006.15437)
* [ ] 2020 [GraphSAINT: GRAPH SAMPLING BASED INDUCTIVE LEARNING METHOD](https://openreview.net/pdf?id=BJe8pkHFwS)
* [ ] 2020 [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/pdf/2004.11198)
* [ ] 2020 [NARS: SCALABLE GRAPH NEURAL NETWORKS FOR HETEROGENEOUS GRAPHS](https://arxiv.org/pdf/2011.09679)
* [ ] 2021 [SuperGAT: HOW TO FIND YOUR FRIENDLY NEIGHBORHOOD: GRAPH ATTENTION DESIGN WITH SELF-SUPERVISION](https://openreview.net/pdf?id=Wi5KUNlqWty)
* [ ] 2021 [R-HGNN: Heterogeneous Graph Representation Learning with Relation Awareness](https://arxiv.org/pdf/2105.11122)


#### Self-Supervised/Pre-Training

* [ ] 2020 [Self-Supervised Graph Representation Learning via Global Context Prediction](https://arxiv.org/pdf/2003.01604)
* [ ] 2020 [When Does Self-Supervision Help Graph Convolutional Networks?](http://proceedings.mlr.press/v119/you20a/you20a.pdf)
* [ ] 2020 [Strategies for Pre-Training Graph Neural Networks](https://www.openreview.net/pdf?id=HJlWWJSFDH)
* [ ] 2021 [Self-Supervised Learning of Contextual Embeddings for Link Prediction in Heterogeneous Networks](https://arxiv.org/pdf/2007.11192)
* [ ] 2021 [HeCo: Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning](https://arxiv.org/pdf/2105.09111)

#### Others

* [ ] 2021 [C&S: Combining Label Propagation and Simple Models Out-Performs Graph Neural Networks](https://arxiv.org/pdf/2010.13993)

### Recommendation Algorithms Based on Graph Neural Networks
#### Overview

* [ ] 2020 [A Survey on Knowledge Graph-Based Recommender Systems](https://arxiv.org/pdf/2003.00911)
* [ ] 2020 [Graph Neural Networks in Recommender Systems: A Survey](http://arxiv.org/pdf/2011.02260)

#### Embedding-based Methods

* [ ] 2016 [CKE: Collaborative Knowledge Base Embedding for Recommender Systems](https://www.kdd.org/kdd2016/papers/files/adf0066-zhangA.pdf)
* [ ] 2018 [CFKG: Learning over Knowledge-Base Embeddings for Recommendation](https://arxiv.org/pdf/1803.06540)
* [ ] 2018 [SHINE: Signed Heterogeneous Information Network Embedding for Sentiment Link Prediction](https://arxiv.org/pdf/1712.00732)

#### Path-based Methods

* [ ] 2013 [Hete-MF: Collaborative Filtering with Entity Similarity Regularization in Heterogeneous Information Networks](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=923e16b43dcd26b59f32c71ef366bf70588853f8)
* [ ] 2014 [Hete-CF: Social-Based Collaborative Filtering Recommendation using Heterogeneous Relations](https://arxiv.org/pdf/1412.7610)
* [ ] 2013 [HeteRec: Recommendation in Heterogeneous Information Networks with Implicit User Feedback∗](http://hanj.cs.illinois.edu/pdf/recsys13_xyu.pdf)
* [ ] 2015 [SemRec: Semantic Path based Personalized Recommendation on Weighted Heterogeneous Information Networks](https://papers-gamma.link/static/memory/pdfs/152-Shi_Semantic_Path_Based_Personalized_Recommendation_on_Weighted_HIN_2015.pdf)
* [ ] 2019 [RuleRec: Jointly Learning Explainable Rules for Recommendation with Knowledge Graph](https://arxiv.org/pdf/1903.03714)
* [ ] 2018 [MCRec: Leveraging Meta-path based Context for Top-N Recommendation with A Neural Co-Attention Model](https://dl.acm.org/doi/pdf/10.1145/3219819.3219965)
* [ ] 2018 [RKGE: Recurrent knowledge graph embedding for effective recommendation](https://repository.tudelft.nl/islandora/object/uuid:9a3559e9-27b6-47cd-820d-d7ecc76cbc06/datastream/OBJ/download)

#### Combined Methods of Embedding and Path

* [ ] 2018 [RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems](https://arxiv.org/pdf/1803.03467)
* [ ] 2019 [AKUPM: Attention-Enhanced Knowledge-Aware User Preference Model for Recommendation](https://dl.acm.org/doi/abs/10.1145/3292500.3330705)
* [ ] 2019 [KGCN: Knowledge Graph Convolutional Networks for Recommender Systems](https://arxiv.org/pdf/1904.12575)
* [ ] 2019 [KGAT: KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/pdf/1905.07854)
* [ ] 2019 [KNI: An End-to-End Neighborhood-based Interaction Model for Knowledge-enhanced Recommendation](https://arxiv.org/pdf/1908.04032)

## Model Replication

### Heterogeneous Graph Representation Learning

* [ ] [GCN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/gcn)
* [ ] [R-GCN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/rgcn)
* [ ] [GAT](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/gat)
* [ ] [HetGNN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/hetgnn)
* [ ] [HAN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/han)
* [ ] [MAGNN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/magnn)
* [ ] [HGT](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/hgt)
* [ ] [metapath2vec](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/metapath2vec)
* [ ] [SIGN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/sign)
* [ ] [HGConv](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/hgconv)
* [ ] [SuperGAT](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/supergat)
* [ ] [R-HGNN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/rhgnn)
* [ ] [C&S](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/cs)
* [ ] [HeCo](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/heco)

### Recommendation Algorithms Based on Graph Neural Networks

* [ ] CKE
* [ ] RippleNet
* [ ] KGCN
* [ ] KGAT

## Detailed Plan

* [ ] 2024/03/10 - 2024/03/17
	* [ ] Collect datasets (DBLP, Microsoft OAG)
	* [ ] Study the structure of the datasets
	* [ ] Read the survey articles

* [ ] 2024/03/18 - 2024/03/24
	* [ ] Explore and visualize datasets
	* [ ] Crawl data for validation set
	* [ ] Filter data for computer science topic only
	* [ ] Explore preprocessing techniques