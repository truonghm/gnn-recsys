# Literature review

This review is a summary of the article [Scientific Paper Recommendation Systems: a Literature Review of recent Publications](https://arxiv.org/pdf/2201.00682). The article discusses many different papers focusing on scientific paper recommendation systems, but this summary will only focus on papers that utilize graph structure, which total to 25 papers published between (inclusive) 2019 and 2021. The summary will also emphasize papers that discuss non-personalized recommendation systems, i.e. systems that do not use user profiles to make recommendations.

## Methods

### General methodologies

In general, most of the papers follow a loosely similar structure:

1. **Graph Construction**: First, they construct a graph incorporating multiple types of entities (authors, papers, venues, topics) and relationships (citations, co-authorships, topical relevance).

2. **Representation Learning**: Next, a variation or combination of embedding representation (SBERT, LDA, TF-IDF, Word2Vec, or Doc2Vec), random walk/meta-path analysis and/or Graph Neural Network is used. The result is the generation of embeddings that capture the structural and semantic information of the graph.

3. **Recommendation and ranking**: Some form of closeness measurement or grouping is used in this step, such as cosine similarity or clustering methods. The papers are ranked by this measurement, while some are ordered in a path. Results can also be adjusted based on the recency of papers (using a time decay factor) or the popularity of authors.

### Types of graph structures

- Most papers use some form of heterogenous graph strucuture.
- Most build a graph using one of or a combination (see [Ali2020](papers/Ali2020.pdf), [Du2020](papers/Du2020.pdf), [Hua2020](papers/Hua2020.pdf), [Li2021](papers/Li2021.pdf), [Zhang2019](papers/Zhang2019.pdf), [Zhang2020](papers/Zhang2020.pdf)) of the following:
  - Authors ([Hu2020](papers/Hu2020.pdf))
  - Papers ([L2021](papers/L2021.pdf))
  - Venues
  - Labels/Keywords/Topics
  - Citations (directed or undirected graphs). Some papers use co-citations or bibliographi coupling or both. See [Jing2020](papers/Jing2020.pdf), [Kanakia2019](papers/Kanakia2019.pdf), [Kang2021](papers/Kang2021.pdf), [Sakib2020](papers/Sakib2020.pdf), [Tanner2019](papers/Tanner2019.pdf), [Yang2019](papers/Yang2019.pdf), [Shi2020](papers/Shi2020.pdf), [Waheed2019](papers/Waheed2019.pdf).

### Meta-paths

- [Hua2020](papers/Hua2020.pdf) construct author-paper-author and author-paper-venue-paper-author paths by applying beam search. Papers on the most similar paths are recommended to users.
- [Li2021](papers/Li2021.pdf) construct meta-paths of a max length between users and papers and use random walk on these paths.
- [Ma2018](papers/Ma2018.pdf) use meta-paths to measure the proximity between nodes in a graph.


### Random walk

- [Du2020](papers/Du2020.pdf) and [Manju2020](papers/Manju2020.pdf) use random walk on heterogeneous graphs.
- [Hua2020](papers/Hua2020.pdf), [Li2021](papers/Li2021.pdf), and [Ma2018](papers/Ma2018.pdf) use random walk to identify or determine the proximity between meta-paths.

### Graph Neural Networks & Graph Embeddings

- Graph Neural Network: [L2021](papers/L2021.pdf), [Tang2021](papers/Tang2021.pdf), [Yu2021](papers/Yu2021.pdf), however the 2 latter papers are for personalized systems.
- Heterogeneous graph embeddings: [Du2020](papers/Du2020.pdf), [Hua2020](papers/Hua2020.pdf), [Kong2018](papers/Kong2018.pdf)


### Cosine Similarity

Many papers use some form of cosine similarity in their recommendation systems. 

1. The primary application of cosine similarity is to calculate the similarity between papers:

  - Unspecified word or vector representations of papers: [Kong2018](papers/Kong2018.pdf)
  - Papers' key phrases: [Kang2021](papers/Kang2021.pdf)
  - Vector space model variants, such as TF-IDF vectors in [Hua2020](papers/Hua2020.pdf)
  - More advanced embeddings techniques:  SBERT embeddings in [Ali2020](papers/Ali2020.pdf) and  NPLM representations in [Du2020](papers/Du2020.pdf) 

2. Other applications include the usage between input keywords and paper clusters ([Yang2019](papers/Yang2019.pdf)) and between nodes in a graph represented by their neighbouring nodes ([Zhang2019](papers/Zhang2019.pdf) and [Zhang2020](papers/Zhang2020.pdf)).

### Other methods

- [Liu2020](papers/Liu2020.pdf) uses Steiner trees construction
- [Wang2020](papers/Wang2020.pdf) uses BFS
- [Yang2019](papers/Yang2019.pdf) uses clustering and calculation of closeness degree

## Paper summary

Below is the list of papers that were included in the review that focus on graph-based recommendation systems.


* [ ] [Ali2020](papers/Ali2020.pdf) construct PR-HNE, a personalised probabilistic paper recommendation model based on a joint representation of authors and publications. They utilise graph information such as citations as well as coauthorships, venue information and topical relevance to suggest papers. They apply SBERT and LDA to represent author embeddings and topic embeddings respectively.

* [ ] [Du2020](papers/Du2020.pdf)  introduce HNPR, a heterogeneous network method using two different graphs. The approach incorporates citation information, co-author relations and research areas of publications. They apply random walk on the networks to generate vector representations of papers.

* [ ] [Hu2020](papers/Hu2020.pdf) present ADRCR, a paper recommendation approach incorporating author-author and author-paper citation relationships as well as authors' and papers' authoritativeness. A network is built which uses citation information as weights. Matrix decomposition helps learning the model.

* [X] [Hua2020](papers/Hua2020.pdf):
  1. Use text learning (TF-IDF) and network learning (random walk & meta-path) to define relatedness between papers. Aggregate the two to create a united relatedness measurement that includes both semantic similarity and network closeness. Also incorporate time decay to avoid overemphasizing old important articles.
  2.  Use a a two-stage beam search component to generate an ordered path.

* [ ] [Kanakia2019](papers/Kanakia2019.pdf) build their approach upon the MAG dataset and strive to overcome the common problems of scalability and cold-start. They combine TF-IDF and Word2Vec representations of the content with co-citations of papers to compute recommendations. Speedup is achieved by comparing papers to clusters of papers instead of all other single papers.

* [ ] [Kang2021](papers/Kang2021.pdf) crawl full texts of papers from CiteSeer and construct citation graphs to determine candidate papers. Then they compute a combination of section-based citation and key phrase similarity to rank recommendations.

* [ ] [Kong2018](papers/Kong2018.pdf) present VOPRec, a model combining textual components in form of Doc2vec and Paper2Vec paper representations with citation network information in form of Struc2vec. Those networks of papers connect the most similar publications based on text and structure. Random walk on these graphs contributes to the goal of learning vector representations.

* [ ] [L2021](papers/L2021.pdf) base their recommendation on lately accessed papers of authors as they assume future accessed papers are similar to recently seen ones. They utilise a sliding window to generate sequences of papers, on those they construct a GNN to aggregate neighbouring papers to identify authors' interests.

* [ ] [Li2021](papers/Li2021.pdf)  present HNTA a paper recommendation method utilising heterogeneous networks and changing scholar interests. Paper similarities are calculated with Word2Vec representations of words recommended for each paper. Changing scholar interest is modelled with help of an exponential time decay function on word vectors.

* [ ] [Liu2020](papers/Liu2020.pdf) propose an approach utilising numbers of citations (author popularity) and relationships between publications in an undirected citation graph. They compute Steiner trees to identify the sets of papers to recommend.

* [ ] [Ma2018](papers/Ma2018.pdf) introduce HIPRec, a paper recommendation approach on heterogeneous networks of authors, papers, venues and topics specialised on new publications. They use the most interesting meta-paths to construct significant meta-paths. With these paths and features from these paths they train a model to identify new papers fitting users. 

* [ ] [Ma2019](papers/Ma2019.pdf) propose HGRec, a heterogeneous graph representation learning-based model working on the same network as [Ma2018](papers/Ma2018.pdf). They use meta-path-based features and Doc2Vec paper embeddings to learn the node embeddings in the network.

* [ ] [Manju2020](papers/Manju2020.pdf) attempt to solve the cold-start problem with their paper recommendation approach coding social interactions as well as topical relevance into a heterogeneous graph. They incorporate believe propagation into the network and compute recommendations by applying random walk.

* [ ] [Sakib2020](papers/Sakib2020.pdf)  present a paper recommendation approach utilising second-level citation information and citation context. They strive to not rely on user profiles in the paper recommendation process. Instead they measure similarity of candidate papers to an input paper based on co-occurred or co-occurring papers.

* [ ] [Shi2020](papers/Shi2020.pdf) present AMHG, an approach utilising a multilayer perceptron. They also construct a multilevel citation network as described before with added author relations. Here they additionally utilise vector representations of publications and recency.

* [ ] [Tang2021](papers/Tang2021.pdf) introduce CGPrec, a content-based and knowledge graph-based paper recommendation system. They focus on users’ sparse interaction history with papers and strive to predict papers on which users are likely to click. They utilise Word2Vec and a Double Convolutional Neural Network to emulate users’ preferences directly from paper content as well as indirectly by using knowledge graphs.

* [ ] [Tanner2019](papers/Tanner2019.pdf) consider relevance and strength of citation relations to weigh the citation network. They fetch citation information from the parsed full texts of papers. On the weighted citation networks they run either weighted co-citation inverse document frequency, weighted bibliographic coupling or random walk with restart to identify the highest scoring papers.

* [ ] [Waheed2019](papers/Waheed2019.pdf) propose CNRN, a recommendation approach using a multilevel citation and authorship network to identify recommendation candidates. From these candidate papers ones to recommend are chosen by combining centrality measures and authors' popularity. 

* [ ] [Wang2020](papers/Wang2020.pdf) introduce a knowledge-aware path recurrent network model. An LSTM mines path information from the knowledge graphs incorporating papers and users. Users are represented by their downloaded, collected and browsed papers, papers are represented by TF-IDF representations of their keywords.

* [ ] [Yang2019](papers/Yang2019.pdf) incorporate the age of papers and impact factors of venues as weights in their citation network-based approach named PubTeller. Papers are
clustered by topic, the most popular ones from the clusters most similar to the query terms are recommendation candidates. In this approach, LDA and TF-IDF are used to represent publications.

* [ ] [Zhang2019](papers/Zhang2019.pdf) propose W-Rank, a general approach weighting edges in a heterogeneous author, paper and venue graph by incorporating citation relevance and author contribution. They apply their method on paper recommendation. Network- (via citations) and semantic-based (via AWD) similarity between papers is combined for weighting edges between papers, harmonic counting defines weights of edges between authors and papers. A HITS-inspired algorithm computes the final authority scores.

* [ ] [Zhang2020](papers/Zhang2020.pdf) strive to emulate a human expert recommending papers. They construct a heterogeneous network with authors, papers, venues and citations. Citation weights are determined by semantic and network-level similarity of papers. Lastly, recommendation candidates are re-ranked while combining the weighted heterogeneous network and recency of papers.

## Evaluation

### Supervised Labelling

In terms of supervised labels, ***most papers use datasets with no explicit labels*** (dblp, AMiner, ACM, MAG, CiteSeerX, etc.). There are several that do have labels, such as:

- Supervised labels provided by human annotators in the form of sets of papers, which researchers found relevant for themselves, such as the [SPRD based datasets](https://scholarbank.nus.edu.sg/handle/10635/146027).
- Interaction data from users, such as clicks, likes, downloads, etc. such as the [Citeulike datasets](https://github.com/js05212/citeulike-a).
- Other labelled datasets: [PRSDDataset](https://sites.google.com/site/tinhuynhuit/dataset), [RARD II](https://arxiv.org/abs/1807.06918).

### Measurement

The performance of a paper recommendation system can be quantified by measuring how well a target value has been approximated by the recommended publications. Specifically, there are 3 areas of evaluation:

- **Revelency**: Method used to evaluate the relevance of the recommended papers, including:
  - Human evaluation
  - Evaluation against a dataset (such as an implicit feedback dataset)

- **Target value**: describe the entities which the approach tried to approximate, this could be:
  - Users' activities on the papers (such as clicks, views, downloads, etc.)
  - Automatic derivation: Number of citations, papers with identical references, degree-of-relevancy, keywords extracted from papers, papers containing the query keywords in the optimal Steiner tree, neighbouring (cited and referencing) papers, included keywords, the classification tag, etc.

- **Measurement**: The following metrics are often used (most papers report at least 2 measures):
  - Precision
  - Recall
  - F1
  - nDCG (Normalized Discounted Cumulative Gain)
  - MRR (Mean Reciprocal Rank)
  - MAP (Mean Average Precision)

### Baseline methods

Taken from [Hua2020](papers/Hua2020.pdf), the following are some of the baseline methods used in the papers:

- DeepWalk [DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652)
- Node2Vec [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653)
- LDA
- LSI
- PathSim
- HeteSim
- PageRank
- PAPR


## Annex

Below is a list of related methods found in the papers (and some other papers outside of the literature review) that were not included in the summary above. They are graph-based neural network models, graph representation learning methods, and graph-based recommendation algorithms.

### Graph-based Neural Network and Graph Representation Learning

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

### Recommendation Algorithms Based on Graph Neural Networks

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