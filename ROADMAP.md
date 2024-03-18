# Project Roadmap

## Overall Plan

* [ ] 2024/03: Collect resoureces (papers, datasets) and study foundational materials
* [ ] 2024/03: Explore and visualize datasets, crawl data, filter data for computer science topic only
* [ ] 2024/04: Start reading related papers and define clearly the structure of the project (preprocessing, training, evaluation, etc)
* [ ] 2024/04: Start implementing existing GNN models and recommendation algorithms

## Paper Reading

### Graph Neural Networks and Graph Representation Learning

#### Books & Tutorials

* [ ] [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf)
* [ ] [Deep Graph Learning: Foundations, Advances and Applications](https://ai.tencent.com/ailab/ml/KDD-Deep-Graph-Learning.html)

#### Surveys

* [ ] 2018 [A Comprehensive Survey of Graph Embedding: Problems, Techniques and Applications](https://ieeexplore.ieee.org/document/8294302)
* [ ] 2019 [A Comprehensive Survey on Graph Neural Networks](https://ieeexplore.ieee.org/document/9046288)
* [ ] 2019 [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/pdf/1812.08434.pdf)
* [ ] 2017 [Representation Learning on Graphs: Methods and Applications](https://arxiv.org/pdf/1709.05584.pdf)

### Frameworks/Tools

* [ ] [DGL: Deep Graph Library](https://github.com/dmlc/dgl)

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
* [ ] 2023 [(RKGCN: Ripple Knowledge Graph Convolutional Networks For Recommendation Systems](https://link.springer.com/content/pdf/10.1007/s11633-023-1440-x.pdf)

## Dataset

* [ ] [DBLP](https://dblp.uni-trier.de/)
* [ ] [Microsoft OAG](https://www.microsoft.com/en-us/research/project/open-academic-graph/)

## Model Replication

### Recommendation Algorithms Based on Graph Neural Networks

* [ ] CKE
* [ ] RippleNet
* [ ] KGCN
* [ ] KGAT
* [ ] RKGCN

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
	* [ ] Learning [DGL](https://www.dgl.ai/) (Deep Graph Library)