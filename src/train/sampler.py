from typing import Dict, Iterator, List, Optional, Tuple

import dgl
import torch
from torch.utils.data import Sampler


class ContrastiveSampler(Sampler):
    """Sampler for contrastive learning on heterogeneous graphs.

    This sampler generates:
    1. Anchor papers (query nodes)
    2. Positive examples (papers sharing metadata)
    3. Negative examples (randomly sampled papers)
    """

    def __init__(
        self,
        g: dgl.DGLGraph,
        batch_size: int,
        num_negative: int = 5,
        share_metadata: Optional[List[Tuple[str, str, str]]] = None,
    ):
        """Initialize the sampler.

        Parameters
        ----------
        g : dgl.DGLGraph
            The heterogeneous graph
        batch_size : int
            Number of anchor nodes per batch
        num_negative : int
            Number of negative samples per anchor
        share_metadata : List[Tuple[str, str, str]], optional
            Edge types to consider for positive sampling
            e.g. [("paper", "has_field", "field"), ("paper", "published_at", "venue")]
        """
        self.g = g
        self.batch_size = batch_size
        self.num_negative = num_negative
        self.share_metadata = share_metadata or []

        # Pre-compute positive pairs based on shared metadata
        self.pos_pairs = self._find_positive_pairs()

        # Total number of papers
        self.num_papers = g.num_nodes("paper")

    def _find_positive_pairs(self) -> Dict[int, List[int]]:
        """Find papers that share metadata (fields, venues, etc).

        Returns
        -------
        Dict[int, List[int]]
            Mapping from paper ID to list of positive example IDs
        """
        pos_pairs: Dict[int, List[int]] = {}

        for src_type, edge_type, dst_type in self.share_metadata:
            # Get paper pairs that share metadata
            src_dst_pairs: List[Tuple[int, int]] = []
            for etype in [(src_type, edge_type, dst_type), (dst_type, edge_type + "_rev", src_type)]:
                if etype in self.g.canonical_etypes:
                    src, dst = self.g.edges(etype=etype)
                    src_dst_pairs.extend(zip(src.tolist(), dst.tolist()))

            # Group papers by shared metadata
            metadata_groups: Dict[int, List[int]] = {}
            for src, dst in src_dst_pairs:
                if dst not in metadata_groups:
                    metadata_groups[dst] = []
                metadata_groups[dst].append(src)

            # Create positive pairs
            for papers in metadata_groups.values():
                if len(papers) > 1:  # Only consider groups with multiple papers
                    for i in papers:
                        if i not in pos_pairs:
                            pos_pairs[i] = []
                        pos_pairs[i].extend([j for j in papers if j != i])

        return pos_pairs

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Generate batches of samples.

        Each batch contains:
        - Anchor paper indices
        - Positive example indices for each anchor
        - Negative example indices for each anchor
        """
        # Papers that have positive examples
        valid_anchors = list(self.pos_pairs.keys())

        # Shuffle anchors
        perm = torch.randperm(len(valid_anchors))

        for idx in range(0, len(valid_anchors), self.batch_size):
            # Get batch of anchor papers
            batch_idx = perm[idx : min(idx + self.batch_size, len(valid_anchors))]
            anchors = [valid_anchors[i] for i in batch_idx]

            # Get positive examples for each anchor
            positives = [self.pos_pairs[a] for a in anchors]

            # Sample negative examples for each anchor
            negatives = []
            for a in anchors:
                # Exclude anchor and its positives from negative sampling
                exclude = set([a] + self.pos_pairs[a])
                neg = torch.randperm(self.num_papers)[: self.num_negative]
                # Resample any negatives that are actually positives
                mask = torch.tensor([n.item() in exclude for n in neg])
                while mask.any():
                    neg[mask] = torch.randperm(self.num_papers)[: mask.sum()]
                    mask = torch.tensor([n.item() in exclude for n in neg])
                negatives.append(neg.tolist())

            yield {
                "anchor": torch.tensor(anchors),
                "positive": torch.tensor([p[0] for p in positives]),  # Take first positive for each anchor
                "negative": torch.tensor(negatives),
            }

    def __len__(self) -> int:
        """Number of batches."""
        return (len(self.pos_pairs) + self.batch_size - 1) // self.batch_size
