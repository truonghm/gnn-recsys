import json
import os
from collections import defaultdict
from itertools import chain

import dgl
import dgl.function as fn
import pandas as pd
import torch
from dgl.data import DGLDataset, extract_archive
from dgl.data.utils import load_graphs, save_graphs
from gensim.models import Word2Vec
from src.config import settings

DATA_DIR = settings.DATA_DIR


def iter_json(filename):
    with open(filename, encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


class OAGCSDataset(DGLDataset):
    """Microsoft OAG 2.1 dataset, filtered to Computer Science topic.

    This dataset contains a single heterogeneous graph with the following statistics:

    Vertices
    --------
    - Authors: 2,248,205
    - Papers: 1,852,225
    - Venues: 11,177
    - Institutions: 13,747
    - Fields: 120,992

    Edges
    -----
    - Author-writes->Paper: 6,349,317
    - Paper-published_at->Venue: 1,852,225
    - Paper-has_field->Field: 17,250,107
    - Paper-cites->Paper: 9,194,781
    - Author-affiliated_with->Institution: 1,726,212

    Attributes
    ----------
    Paper vertices:
        feat : ndarray of shape (N_paper, 128)
            Pre-trained title and abstract word vectors.
        year : ndarray of shape (N_paper,)
            Publication year (2010-2021).
        citation : ndarray of shape (N_paper,)
            Number of citations.
        Note: Does not include labels.

    Field vertices:
        feat : ndarray of shape (N_field, 128)
            Pre-trained field vectors.

    Writes edges:
        order : ndarray of shape (N_writes,)
            Author order (starting from 1).
    """

    def __init__(self, **kwargs):
        super().__init__("oag-cs", "na", **kwargs)

    def download(self):
        zip_file_path = os.path.join(self.raw_dir, "oag-cs.zip")
        if not os.path.exists(zip_file_path):
            for root, dirs, files in os.walk(self.raw_dir):
                for d in dirs:
                    if d.startswith("oag-cs_"):
                        os.rename(os.path.join(root, d), self.raw_path)
                        return
            else:
                raise FileNotFoundError(f"A zip file called `oag-cs.zip` is required in {self.raw_dir}")
        extract_archive(zip_file_path, self.raw_path)

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + "_dgl_graph.bin"), [self.g])

    def load(self):
        self.g = load_graphs(os.path.join(self.save_path, self.name + "_dgl_graph.bin"))[0][0]

    def process(self):
        self._vid_map = self._read_venues()  # {original id: vertex id}
        self._oid_map = self._read_institutions()  # {original id: vertex id}
        self._fid_map = self._read_fields()  # {field name: vertex id}
        self._aid_map, author_inst = self._read_authors()  # {original id: vertex id}, R(aid, oid)
        # PA(pid, aid), PV(pid, vid), PF(pid, fid), PP(pid, rid), [year], [citation count]
        paper_author, paper_venue, paper_field, paper_ref, paper_year, paper_citation = self._read_papers()
        self.g = self._build_graph(
            paper_author, paper_venue, paper_field, paper_ref, author_inst, paper_year, paper_citation
        )
        self.g = self._build_graph(
            paper_author, paper_venue, paper_field, paper_ref, author_inst, paper_year, paper_citation
        )
        self.g = add_reverse_edges(self.g)
        self._add_similarity_edges()

    def _add_similarity_edges(self):
        """
        Adds similarity edges between papers based on shared authors, venues, or fields.

        Logic:
        1. Extract existing edges for author, venue, and field relationships.
        2. Create mappings of authors, venues, and fields to their associated papers.
        3. For each group of papers sharing an author, venue, or field, create
        bidirectional similarity edges between all pairs of papers in the group.
        4. Add these new similarity edges to the graph.

        This method enhances the graph structure for more effective paper recommendations.
        """
        print("Adding similarity edges...")
        paper_author = self.g.edges(etype="writes", form="uv")
        paper_venue = self.g.edges(etype="published_at", form="uv")
        paper_field = self.g.edges(etype="has_field", form="uv")

        # Create dictionaries to store paper relationships
        author_papers = defaultdict(set)
        venue_papers = defaultdict(set)
        field_papers = defaultdict(set)

        # Populate the dictionaries
        for p, a in zip(*paper_author):
            author_papers[a.item()].add(p.item())
        for p, v in zip(*paper_venue):
            venue_papers[v.item()].add(p.item())
        for p, f in zip(*paper_field):
            field_papers[f.item()].add(p.item())

        # Create similarity edges
        similarity_edges = []
        for papers in chain(author_papers.values(), venue_papers.values(), field_papers.values()):
            papers = list(papers)
            for i in range(len(papers)):
                for j in range(i + 1, len(papers)):
                    similarity_edges.append((papers[i], papers[j]))
                    similarity_edges.append((papers[j], papers[i]))

        # Add similarity edges to the graph
        src, dst = zip(*similarity_edges)
        self.g.add_edges(src, dst, etype="similar")

        print(f"Added {len(similarity_edges)} similarity edges")

    def get_paper_features(self):
        return self.g.nodes["paper"].data["feat"]

    def _iter_json(self, filename):
        yield from iter_json(os.path.join(self.raw_path, filename))

    def _read_venues(self):
        print("Reading journal data...")
        # Line number = Index = Vertex id
        return {v["id"]: i for i, v in enumerate(self._iter_json("mag_venues.txt"))}

    def _read_institutions(self):
        print("Reading institution data...")
        return {o["id"]: i for i, o in enumerate(self._iter_json("mag_institutions.txt"))}

    def _read_fields(self):
        print("Reading field data...")
        return {f["name"]: f["id"] for f in self._iter_json("mag_fields.txt")}

    def _read_authors(self):
        print("Reading scholar data...")
        author_id_map, author_inst = {}, []
        for i, a in enumerate(self._iter_json("mag_authors.txt")):
            author_id_map[a["id"]] = i
            if a["org"] is not None:
                author_inst.append([i, self._oid_map[a["org"]]])
        return author_id_map, pd.DataFrame(author_inst, columns=["aid", "oid"])

    def _read_papers(self):
        print("Reading paper data...")
        paper_id_map, paper_author, paper_venue, paper_field = {}, [], [], []
        paper_year, paper_citation = [], []
        for i, p in enumerate(self._iter_json("mag_papers.txt")):
            paper_id_map[p["id"]] = i
            paper_author.extend([i, self._aid_map[a], r + 1] for r, a in enumerate(p["authors"]))
            paper_venue.append([i, self._vid_map[p["venue"]]])
            paper_field.extend([i, self._fid_map[f]] for f in p["fos"] if f in self._fid_map)
            paper_year.append(p["year"])
            paper_citation.append(p["n_citation"])

        paper_ref = []
        for i, p in enumerate(self._iter_json("mag_papers.txt")):
            paper_ref.extend([i, paper_id_map[r]] for r in p["references"] if r in paper_id_map)
        return (
            pd.DataFrame(paper_author, columns=["pid", "aid", "order"]).drop_duplicates(subset=["pid", "aid"]),
            pd.DataFrame(paper_venue, columns=["pid", "vid"]),
            pd.DataFrame(paper_field, columns=["pid", "fid"]),
            pd.DataFrame(paper_ref, columns=["pid", "rid"]),
            paper_year,
            paper_citation,
        )

    def _build_graph(self, paper_author, paper_venue, paper_field, paper_ref, author_inst, paper_year, paper_citation):
        print("Constructing the heterogeneous graph...")
        pa_p, pa_a = paper_author["pid"].to_list(), paper_author["aid"].to_list()
        pv_p, pv_v = paper_venue["pid"].to_list(), paper_venue["vid"].to_list()
        pf_p, pf_f = paper_field["pid"].to_list(), paper_field["fid"].to_list()
        pp_p, pp_r = paper_ref["pid"].to_list(), paper_ref["rid"].to_list()
        ai_a, ai_i = author_inst["aid"].to_list(), author_inst["oid"].to_list()
        g = dgl.heterograph(
            {
                ("author", "writes", "paper"): (pa_a, pa_p),
                ("paper", "published_at", "venue"): (pv_p, pv_v),
                ("paper", "has_field", "field"): (pf_p, pf_f),
                ("paper", "cites", "paper"): (pp_p, pp_r),
                ("author", "affiliated_with", "institution"): (ai_a, ai_i),
            }
        )
        g.nodes["paper"].data["feat"] = torch.load(os.path.join(self.raw_path, "paper_feat.pkl"))
        g.nodes["paper"].data["year"] = torch.tensor(paper_year)
        g.nodes["paper"].data["citation"] = torch.tensor(paper_citation)
        g.nodes["field"].data["feat"] = torch.load(os.path.join(self.raw_path, "field_feat.pkl"))
        g.edges["writes"].data["order"] = torch.tensor(paper_author["order"].to_list())
        return g

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, self.name + "_dgl_graph.bin"))

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("This dataset has only one graph")
        return self.g

    def __len__(self):
        return 1


def load_data(device="cpu", add_reverse_edge=True):
    """Load the OAG-CS dataset

    Parameters
    ----------
    device : torch.device, optional
        Device to move the graph and data to, default is CPU
    add_reverse_edge : bool, optional
        Whether to add reverse edges, default is True

    Returns
    -------
    dataset : OAGCSDataset
        The loaded dataset
    g : DGLGraph
        The graph
    features : torch.Tensor
        Node features
    labels : None
        Labels (None for recommendation task)
    predict_ntype : str
        Node type for prediction
    train_idx : torch.Tensor
        Indices for training set
    val_idx : torch.Tensor
        Indices for validation set
    test_idx : torch.Tensor
        Indices for test set
    evaluator : None
        Evaluator (None for this dataset)
    """
    data = OAGCSDataset()
    g = data[0]

    if add_reverse_edge:
        g = add_reverse_edges(g)

    g = g.to(device)
    features = g.nodes["paper"].data["feat"]

    # For recommendation task, we don't need labels
    labels = None

    # Create train/val/test split for papers
    num_papers = g.num_nodes("paper")
    perm = torch.randperm(num_papers)
    train_idx = perm[: int(0.8 * num_papers)]
    val_idx = perm[int(0.8 * num_papers) : int(0.9 * num_papers)]
    test_idx = perm[int(0.9 * num_papers) :]

    return data, g, features, labels, train_idx, val_idx, test_idx, None


def add_reverse_edges(g, reverse_self=True):
    """Add reverse edges to each type of edge in the heterogeneous graph, returning a new heterogeneous graph

    Parameters
    ----------
    g : DGLGraph
        Heterogeneous graph
    reverse_self : bool, optional
        Whether to add reverse edges when source and destination node types are the same, default is True

    Returns
    -------
    DGLGraph
        Heterogeneous graph with added reverse edges
    """
    data = {}
    for stype, etype, dtype in g.canonical_etypes:
        u, v = g.edges(etype=(stype, etype, dtype))
        data[(stype, etype, dtype)] = u, v
        if stype != dtype or reverse_self:
            data[(dtype, etype + "_rev", stype)] = v, u
    new_g = dgl.heterograph(data, {ntype: g.num_nodes(ntype) for ntype in g.ntypes})
    for ntype in g.ntypes:
        new_g.nodes[ntype].data.update(g.nodes[ntype].data)
    for etype in g.canonical_etypes:
        new_g.edges[etype].data.update(g.edges[etype].data)
    return new_g


def one_hot_node_feat(g):
    for ntype in g.ntypes:
        if "feat" not in g.nodes[ntype].data:
            g.nodes[ntype].data["feat"] = torch.eye(g.num_nodes(ntype), device=g.device)


def average_node_feat(g):
    """For nodes without input features in the ogbn-mag dataset, take the average of neighbors"""
    message_func, reduce_func = fn.copy_u("feat", "m"), fn.mean("m", "feat")
    g.multi_update_all({"writes_rev": (message_func, reduce_func), "has_topic": (message_func, reduce_func)}, "sum")
    g.multi_update_all({"affiliated_with": (message_func, reduce_func)}, "sum")


def load_pretrained_node_embed(g, node_embed_path, concat=False, use_raw_id=False):
    """Load pretrained node features for nodes without input features

    Parameters
    ----------
    g : DGLGraph
        Heterogeneous graph
    node_embed_path : str
        Path to the pretrained word2vec model
    concat : bool, optional
        If True, concatenate pretrained features with original input features
    use_raw_id : bool, optional
        Use original node id

    """
    model = Word2Vec.load(node_embed_path)
    for ntype in g.ntypes:
        if use_raw_id:
            embed = torch.from_numpy(model.wv[[f"{ntype}_{i}" for i in g.nodes[ntype].data[dgl.NID].tolist()]]).to(
                g.device
            )
        else:
            embed = torch.from_numpy(model.wv[[f"{ntype}_{i}" for i in range(g.num_nodes(ntype))]]).to(g.device)
        if "feat" in g.nodes[ntype].data:
            if concat:
                g.nodes[ntype].data["feat"] = torch.cat([g.nodes[ntype].data["feat"], embed], dim=1)
        else:
            g.nodes[ntype].data["feat"] = embed


def add_node_feat(g, method, node_embed_path=None, concat=False, use_raw_id=False):
    """Add input features for nodes without input features

    Parameters
    ----------
    g : DGLGraph
        Heterogeneous graph
    method : str
        Method to add features: 'one-hot', 'average' (only for ogbn-mag dataset), 'pretrained'
    node_embed_path : str, optional
        Path to the pretrained word2vec model
    concat : bool, optional
        If True, concatenate pretrained features with original input features
    use_raw_id : bool, optional
        Use original node id

    Raises
    ------
    ValueError
        If an unknown method is provided
    """
    if method == "one-hot":
        one_hot_node_feat(g)
    elif method == "average":
        average_node_feat(g)
    elif method == "pretrained":
        load_pretrained_node_embed(g, node_embed_path, concat, use_raw_id)
    else:
        raise ValueError(f"add_node_feat: Unknown method {method}")
