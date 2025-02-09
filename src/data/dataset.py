from pathlib import Path
from typing import Dict, Optional

import dgl
import torch
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs

from src.config import settings
from src.data.transforms import add_reverse_edges, add_similarity_edges
from src.data.utils import iter_json


class OAGCSDataset(DGLDataset):
    """Microsoft OAG CS dataset implementation.

    A heterogeneous graph dataset containing computer science papers and related entities.

    Statistics
    ----------
    Vertices:
        - Papers: ~1.8M
        - Authors: ~2.2M
        - Venues: ~11K
        - Institutions: ~13K
        - Fields: ~120K

    Edges:
        - Author-writes->Paper
        - Paper-published_at->Venue
        - Paper-has_field->Field
        - Paper-cites->Paper
        - Author-affiliated_with->Institution

    Features
    --------
    Paper vertices:
        - feat: Pre-trained title and abstract embeddings (dim=768)
        - year: Publication year
        - citation_count: Number of citations

    Field vertices:
        - feat: Pre-trained field embeddings (dim=768)

    Writes edges:
        - order: Author order in paper (1-based)
    """

    def __init__(self, raw_dir: Optional[str] = None, force_reload: bool = False):
        """Initialize the dataset.

        Parameters
        ----------
        raw_dir : str, optional
            Directory to store the raw data files. If None, uses default from config.
        force_reload : bool, optional
            Whether to reload the dataset from raw files.
        """
        self.raw_path = Path(raw_dir) if raw_dir else settings.data.raw_data_dir / "cs"
        self.processed_path = settings.data.processed_data_dir / "cs"
        self.cache_path = settings.data.cache_dir / "cs"

        # Create directories if they don't exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        super().__init__(
            name="oag-cs",
            raw_dir=str(self.raw_path),
            save_dir=str(self.processed_path),
            force_reload=force_reload,
        )

    def download(self):
        """Download is not needed as data should be pre-extracted."""
        pass

    def process(self):
        """Process the raw data into DGL graph."""
        print("Processing OAG CS dataset...")

        # Load all entity mappings
        self._vid_map = self._read_venues()  # venue id -> vertex id
        self._oid_map = self._read_institutions()  # org id -> vertex id
        self._fid_map = self._read_fields()  # field name -> vertex id
        self._aid_map = self._read_authors()  # author id -> vertex id

        # Load paper relations and features
        paper_data = self._read_papers()

        # Construct heterogeneous graph
        self.graph = self._build_graph(paper_data)

        # Add reverse edges and similarity edges
        self.graph = add_reverse_edges(self.graph)
        self.graph = add_similarity_edges(self.graph)

    def _read_venues(self) -> Dict[int, int]:
        """Read venue data and create mapping."""
        print("Reading venues...")
        return {v["id"]: i for i, v in enumerate(self._iter_json("mag_venues.txt"))}

    def _read_institutions(self) -> Dict[int, int]:
        """Read institution data and create mapping."""
        print("Reading institutions...")
        return {o["id"]: i for i, o in enumerate(self._iter_json("mag_institutions.txt"))}

    def _read_fields(self) -> Dict[str, int]:
        """Read field data and create mapping."""
        print("Reading fields...")
        return {f["name"]: f["id"] for f in self._iter_json("mag_fields.txt")}

    def _read_authors(self) -> Dict[int, int]:
        """Read author data and create mapping."""
        print("Reading authors...")
        author_map = {}
        edges = []
        for i, a in enumerate(self._iter_json("mag_authors.txt")):
            author_map[a["id"]] = i
            if a["org"] is not None:
                edges.append((i, self._oid_map[a["org"]]))
        self._author_inst_edges = edges
        return author_map

    def _read_papers(self) -> Dict:
        """Read paper data and relations."""
        print("Reading papers...")
        paper_data = {
            "paper_author": [],  # (pid, aid, order)
            "paper_venue": [],  # (pid, vid)
            "paper_field": [],  # (pid, fid)
            "paper_ref": [],  # (pid, ref_pid)
            "paper_year": [],  # year
            "paper_citation": [],  # citation count
            "paper_feat": [],  # paper features
        }

        for i, p in enumerate(self._iter_json("mag_papers.txt")):
            # Store paper features and metadata
            paper_data["paper_year"].append(p["year"])
            paper_data["paper_citation"].append(p.get("n_citation", 0))

            # Store relations
            for order, aid in enumerate(p["authors"]):
                paper_data["paper_author"].append((i, self._aid_map[aid], order + 1))
            paper_data["paper_venue"].append((i, self._vid_map[p["venue"]]))
            for field in p["fos"]:
                if field in self._fid_map:
                    paper_data["paper_field"].append((i, self._fid_map[field]))
            for ref in p["references"]:
                paper_data["paper_ref"].append((i, ref))

        return paper_data

    def _build_graph(self, paper_data: Dict) -> dgl.DGLGraph:
        """Construct heterogeneous graph from processed data."""
        print("Building graph...")

        # Extract edge data
        pa_p, pa_a, pa_order = zip(*paper_data["paper_author"])
        pv_p, pv_v = zip(*paper_data["paper_venue"])
        pf_p, pf_f = zip(*paper_data["paper_field"])
        pp_p, pp_r = zip(*paper_data["paper_ref"])
        ai_a, ai_i = zip(*self._author_inst_edges)

        # Create heterogeneous graph
        graph_data = {
            ("author", "writes", "paper"): (pa_a, pa_p),
            ("paper", "published_at", "venue"): (pv_p, pv_v),
            ("paper", "has_field", "field"): (pf_p, pf_f),
            ("paper", "cites", "paper"): (pp_p, pp_r),
            ("author", "affiliated_with", "institution"): (ai_a, ai_i),
        }

        g = dgl.heterograph(graph_data)

        # Add features
        g.nodes["paper"].data["year"] = torch.tensor(paper_data["paper_year"])
        g.nodes["paper"].data["citation"] = torch.tensor(paper_data["paper_citation"])
        g.edges["writes"].data["order"] = torch.tensor(pa_order)

        return g

    def save(self):
        """Save processed data."""
        save_graphs(str(self.cache_path / f"{self.name}_dgl_graph.bin"), [self.graph])

    def load(self):
        """Load processed data."""
        graphs, _ = load_graphs(str(self.cache_path / f"{self.name}_dgl_graph.bin"))
        self.graph = graphs[0]

    def has_cache(self):
        """Check if processed data exists."""
        return (self.cache_path / f"{self.name}_dgl_graph.bin").exists()

    def _iter_json(self, filename: str):
        """Helper to iterate over JSON lines in a file."""
        return iter_json(self.raw_path / filename)

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        """Get the graph object."""
        return self.graph

    def __len__(self) -> int:
        """Number of graphs in the dataset."""
        return 1
