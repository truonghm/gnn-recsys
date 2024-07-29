import torch
from dgl import NID, DGLGraph
from dgl.dataloading import BlockSampler, Collator


class PaperTripletCollator(Collator):
    def __init__(self, g: DGLGraph, block_sampler: BlockSampler) -> None:
        """Collator for paper triplets

        Parameters
        ----------
        g : DGLGraph
            Heterogeneous graph
        block_sampler : BlockSampler
            Neighbor sampler
        """
        self.g = g
        self.block_sampler = block_sampler

    def collate(self, items: torch.Tensor) -> tuple[torch.Tensor, list]:
        """Construct subgraph based on paper IDs in triplets

        Parameters
        ----------
        items : tensor
            Shape (B, 3), batch of triplets.

        Returns
        -------
        output_nodes : tensor
            Shape (N_dst,), paper node IDs.
        blocks : list of DGLBlock
            Multi-layer blocks.
        """
        seed_nodes = items.flatten().unique()
        blocks = self.block_sampler.sample_blocks(self.g, {"paper": seed_nodes})
        output_nodes = blocks[-1].dstnodes["paper"].data[NID]
        return output_nodes, blocks

    @property
    def dataset(self) -> None:
        return None
