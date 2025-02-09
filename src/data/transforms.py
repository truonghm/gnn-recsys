from collections import defaultdict
from itertools import chain
from typing import Optional, Set, Tuple

import dgl


def add_reverse_edges(g: dgl.DGLGraph, skip_etypes: Optional[Set[Tuple[str, str, str]]] = None) -> dgl.DGLGraph:
    """Add reverse edges to a heterogeneous graph.

    For each edge type (src, edge_type, dst) in the graph, adds corresponding
    reverse edges (dst, edge_type_rev, src) unless the edge type is in skip_etypes.

    Parameters
    ----------
    g : dgl.DGLGraph
        Input heterogeneous graph
    skip_etypes : set of (str, str, str), optional
        Edge types to skip when adding reverse edges

    Returns
    -------
    dgl.DGLGraph
        New graph with added reverse edges
    """
    if skip_etypes is None:
        skip_etypes = set()

    # Collect all edges to add
    new_edges = {}

    # Copy original edges
    for canonical_etype in g.canonical_etypes:
        new_edges[canonical_etype] = g.edges(etype=canonical_etype)

        # Skip if in skip_etypes
        if canonical_etype in skip_etypes:
            continue

        # Unpack edge type
        src_type, edge_type, dst_type = canonical_etype

        # Get edges for this type
        u, v = g.edges(etype=canonical_etype)

        # Add reverse edges with "_rev" suffix
        rev_etype = (dst_type, f"{edge_type}_rev", src_type)
        new_edges[rev_etype] = (v, u)

    # Create new graph with same node types/counts
    node_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    new_g = dgl.heterograph(new_edges, node_dict)

    # Copy node features
    for ntype in g.ntypes:
        for key, feat in g.nodes[ntype].data.items():
            new_g.nodes[ntype].data[key] = feat

    # Copy edge features
    for canonical_etype in g.canonical_etypes:
        for key, feat in g.edges[canonical_etype].data.items():
            new_g.edges[canonical_etype].data[key] = feat

    return new_g


def add_similarity_edges(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """Add similarity edges between papers based on shared metadata.

    Creates edges between papers that share:
    - Authors
    - Venues
    - Fields

    Parameters
    ----------
    g : dgl.DGLGraph
        Input heterogeneous graph

    Returns
    -------
    dgl.DGLGraph
        Graph with added similarity edges
    """
    print("Adding similarity edges...")

    # Get relevant edges
    paper_author = g.edges(etype="writes", form="uv")
    paper_venue = g.edges(etype="published_at", form="uv")
    paper_field = g.edges(etype="has_field", form="uv")

    # Create mappings of entities to their papers
    author_papers = defaultdict(set)
    venue_papers = defaultdict(set)
    field_papers = defaultdict(set)

    # Populate mappings
    for p, a in zip(*paper_author):
        author_papers[a.item()].add(p.item())
    for p, v in zip(*paper_venue):
        venue_papers[v.item()].add(p.item())
    for p, f in zip(*paper_field):
        field_papers[f.item()].add(p.item())

    # Create similarity edges
    similarity_edges = []
    # Iterate through all paper groups that share an entity
    for papers in chain(author_papers.values(), venue_papers.values(), field_papers.values()):
        papers = list(papers)
        # Create edges between all pairs in group
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                similarity_edges.append((papers[i], papers[j]))
                similarity_edges.append((papers[j], papers[i]))  # Add both directions

    # Add similarity edges to graph
    if similarity_edges:  # Only process if we found edges
        src, dst = zip(*similarity_edges)
        new_g = g.clone()
        new_g.add_edges(src, dst, etype=("paper", "similar", "paper"))
        print(f"Added {len(similarity_edges)} similarity edges")
        return new_g

    return g
