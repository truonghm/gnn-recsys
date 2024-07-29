import argparse
import random
import warnings
from typing import List

import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from dgl.utils import to_dgl_context
from src.data.loaders import add_node_feat

# from gnnrec.kgrec.utils import METRICS_STR, calc_metrics, load_rank_data, recall_paper
from src.data.utils import PaperTripletCollator
from src.models.rhgnn import RHGNN
from src.utils import get_device, set_random_seed
from tqdm import tqdm


def sample_paper_triplets(paper_id: int, cited_papers: List[int], num_papers: int, num_triplets: int) -> torch.Tensor:
    """
    Sample triplets for paper-paper recommendation task.

    Parameters
    ----------
    paper_id : int
        Paper ID.
    cited_papers : list of int
        List of papers cited by paper_id.
    num_papers : int
        Total number of papers in the dataset.
    num_triplets : int
        Number of triplets to sample.

    Returns
    -------
    torch.Tensor
        Sampled triplets with shape (num_triplets, 3).
    """
    n = len(cited_papers)
    easy_margin, hard_margin = int(n * 0.2), int(n * 0.05)
    easy_triplets = [(paper_id, cited_papers[i], cited_papers[i + easy_margin]) for i in range(n - easy_margin)]
    hard_triplets = [(paper_id, cited_papers[i], cited_papers[i + hard_margin]) for i in range(n - hard_margin)]
    m = num_triplets - len(easy_triplets) - len(hard_triplets)
    negative_papers = random.sample(range(num_papers), m)
    negative_triplets = [(paper_id, p, n) for p, n in zip(random.choices(cited_papers, k=m), negative_papers)]
    return torch.tensor(easy_triplets + hard_triplets + negative_triplets)


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    g, paper_citations = load_paper_data(device)
    add_node_feat(g, "pretrained", args.node_embed_path, use_raw_id=True)

    sampler = MultiLayerNeighborSampler([args.neighbor_size] * args.num_layers)
    sampler.set_output_context(to_dgl_context(device))
    triplet_collator = PaperTripletCollator(g, sampler)

    model = RHGNN(
        {ntype: g.nodes[ntype].data["feat"].shape[1] for ntype in g.ntypes},
        args.num_hidden,
        args.num_hidden,
        args.num_rel_hidden,
        args.num_rel_hidden,
        args.num_heads,
        g.ntypes,
        g.canonical_etypes,
        "paper",
        args.num_layers,
        args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(paper_citations) * args.epochs, eta_min=args.lr / 100
    )

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for paper_id in tqdm(paper_citations.keys()):
            triplets = sample_paper_triplets(
                paper_id, paper_citations[paper_id], g.num_nodes("paper"), args.num_triplets
            )
            pid, blocks = triplet_collator.collate(triplets.to(device))
            paper_embeds = model(blocks, blocks[0].srcdata["feat"])
            paper_embeds = paper_embeds / paper_embeds.norm(dim=1, keepdim=True)
            pid_map = {p: i for i, p in enumerate(pid.tolist())}
            anchor = paper_embeds[[pid_map[p] for p in triplets[:, 0].tolist()]]
            positive = paper_embeds[[pid_map[p] for p in triplets[:, 1].tolist()]]
            negative = paper_embeds[[pid_map[p] for p in triplets[:, 2].tolist()]]
            loss = F.triplet_margin_loss(anchor, positive, negative, args.margin)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
        print("Epoch {:d} | Loss {:.4f}".format(epoch, sum(losses) / len(losses)))
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            print(
                METRICS_STR.format(
                    *evaluate(model, g, sampler, args.batch_size, device, paper_citations)
                )
            )

    torch.save(model.state_dict(), args.model_save_path)
    print("Model saved to", args.model_save_path)

    paper_embeds = infer(model, g, "paper", args.num_hidden, sampler, args.batch_size, device)
    paper_embed_save_path = DATA_DIR / "paper_embed.pkl"
    torch.save(paper_embeds.cpu(), paper_embed_save_path)
    print("Paper embeddings saved to", paper_embed_save_path)
