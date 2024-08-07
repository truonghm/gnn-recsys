# Project Roadmap

## Detailed Plan

* [X] 2024/03/10 - 2024/03/17
	* [X] Collect datasets (DBLP, Microsoft OAG)
	* [X] Study the structure of the datasets

* [X] 2024/03/18 - 2024/03/24
	* [X] Explore and visualize datasets
	* [X] Filter data for computer science topic only

* [ ] 2024/05/20 - 2024/05/26
	* [X] Crawl data for validation set
	* [ ] Learning [DGL](https://www.dgl.ai/) (Deep Graph Library)
	* [X] Read the survey articles: [Scientific Paper Recommendation Systems: a Literature Review of recent Publications](https://arxiv.org/abs/2201.00682)
	* [X] Finalize prediction task - pick one of:
		* [X] Given a paper, recommend similar papers
		* [ ] Given a query, recommend papers that are relevant to the query

* [ ] 2024/05/27 - 2024/06/09
	* [X] Filter validation data ([RARD II](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/AT4MNE)) to only contain relevant articles and map them to the original dataset (will need to use the implicit item-item rating matrix based on click logs to generate ground truth) 
	* [X] Read the survey articles
	* [X] Summarize the lit review [Scientific Paper Recommendation Systems: a Literature Review of recent Publications](https://arxiv.org/abs/2201.00682), focusing on GNN (is label needed?)

* [ ] 2024/06/10 - 2024/06/16
	* [ ] test some graph algos such as meta-path, random walk, modularity.
	* [ ] find more papers on recsys that use GNN
	* [ ] implement [Hua2020](papers/Hua2020.pdf) and [DeepWalk](https://arxiv.org/pdf/1403.6652)