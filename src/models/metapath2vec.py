import argparse
import logging

import dgl
import torch
from gensim.models import Word2Vec
from src.data.loaders import OAGCSDataset, add_reverse_edges
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def random_walk(g, metapaths, num_walks, walk_length, output_file):
    """
    Perform random walks on a heterogeneous graph according to specified metapaths and save the traces to a file.

    Parameters
    ----------
    g : DGLGraph
        The heterogeneous graph.
    metapaths : dict of str to list of str
        Mapping from starting node type to metapath. Metapath is represented as a list of edge types.
        The starting and ending node types should be the same.
    num_walks : int
        Number of walks per vertex.
    walk_length : int
        Number of times the metapath is repeated.
    output_file : str
        Name of the output file.

    Returns
    -------
    None
    """
    with open(output_file, "w") as f:
        for ntype, metapath in metapaths.items():
            print(ntype)
            loader = DataLoader(torch.arange(g.num_nodes(ntype)), batch_size=200)
            for b in tqdm(loader):
                nodes = torch.repeat_interleave(b, num_walks)
                traces, types = dgl.sampling.random_walk(g, nodes, metapath=metapath * walk_length)
                f.writelines([trace2name(g, trace, types) + "\n" for trace in traces])


def trace2name(g, trace, types):
    return " ".join(g.ntypes[t] + "_" + str(int(n)) for n, t in zip(trace, types) if int(n) >= 0)


def train_random_walk():
    """
    Perform random walks on a heterogeneous graph and save the traces to a file.

    This function sets up the random walk process by parsing command-line arguments,
    loading the dataset, defining metapaths, and then calling the random_walk function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    This function uses argparse to handle command-line arguments:
    --num-walks : int
        Number of walks per vertex (default: 5)
    --walk-length : int
        Number of times the metapath is repeated (default: 16)
    --output-file : str
        Output file name for saving the random walk traces
    """

    parser = argparse.ArgumentParser(description="Metapath-based random walk")
    parser.add_argument("--num-walks", type=int, default=5, help="Number of walks per vertex")
    parser.add_argument("--walk-length", type=int, default=16, help="Number of times the metapath is repeated")
    parser.add_argument("--output-file", type=str, help="Output file name")
    args = parser.parse_args()

    data = OAGCSDataset()
    g = add_reverse_edges(data[0])
    metapaths = {
        "author": ["writes", "published_at", "published_at_rev", "writes_rev"],  # APVPA
        "paper": ["writes_rev", "writes", "published_at", "published_at_rev", "has_field", "has_field_rev"],  # PAPVPFP
        "venue": ["published_at_rev", "writes_rev", "writes", "published_at"],  # VPAPV
        "field": ["has_field_rev", "writes_rev", "writes", "has_field"],  # FPAPF
        "institution": ["affiliated_with_rev", "writes", "writes_rev", "affiliated_with"],  # IAPAI
    }
    random_walk(g, metapaths, args.num_walks, args.walk_length, args.output_file)


def train_work2vec():
    """
    Train a Word2Vec model on a corpus of random walks.

    This function sets up the Word2Vec training process by parsing command-line arguments
    and then training the model on the provided corpus file.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    This function uses argparse to handle command-line arguments:
    --size : int
        Embedding size (default: 128)
    --workers : int
        Number of workers for parallel processing (default: 4)
    --iter : int
        Number of iterations (epochs) over the corpus (default: 10)
    --corpus-file : str
        Path to the corpus file containing the random walks
    --save-path : str
        Path to save the trained Word2Vec model
    """

    parser = argparse.ArgumentParser(description="Train Word2Vec model")
    parser.add_argument("--size", type=int, default=128, help="Embedding size")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--iter", type=int, default=10, help="Number of iterations")
    parser.add_argument("--corpus-file", type=str, help="Corpus file name")
    parser.add_argument("--save-path", type=str, help="Save path")
    args = parser.parse_args()

    model = Word2Vec(
        corpus_file=args.corpus_file, size=args.size, min_count=1, workers=args.workers, sg=1, iter=args.iter
    )
    model.save(args.save_path)
