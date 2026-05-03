"""Build and cache the MovieGraph from Movie objects.

Edge construction uses three independent similarity signals:

1. Genome cosine similarity – cosine distance between 1128-dim relevance
   vectors from the MovieLens genome.  Available for ~14 K movies.
   Weight contribution: 0.50
2. Audience overlap    – Jaccard of high-rating user sets.  Weight: 0.30
3. Genre similarity    – Jaccard of genre label sets.       Weight: 0.20

For movies without genome data, signals (2) and (3) carry the full weight,
normalised so the maximum possible composite score is still 1.0.

All heavy matrix operations use scipy sparse and numpy so the O(n²)
combinations stay in vectorised C code rather than Python loops.

The assembled NetworkX graph is pickled to data/processed/graph_ml32m.pkl.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import (
    GRAPH_CACHE,
    PROCESSED_DIR,
    load_genome,
    load_high_ratings,
)
from src.models import EdgeRelationship, Movie

# Edge signal weights (when all three signals are present)
_W_GENOME = 0.50
_W_AUD = 0.30
_W_GEN = 0.20

# Minimum composite weight to create an edge
MIN_EDGE_WEIGHT = 0.12

# Audience: user must have rated >= this to count as a "fan"
MIN_RATING_FOR_AUDIENCE = 3.5
# Audience: require at least this many shared fans before computing Jaccard
MIN_SHARED_FANS = 5


# ---------------------------------------------------------------------------
# Genome cosine similarity
# ---------------------------------------------------------------------------

def _genome_cosine(
    all_ids: List[int],
    _genome_override: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Return (cosine_matrix, eligible_ids) for movies with genome vectors."""
    result = _genome_override if _genome_override is not None else load_genome()
    if result is None:
        return np.zeros((0, 0), dtype=np.float32), []

    genome_movie_ids, genome_matrix = result
    genome_id_set = set(genome_movie_ids.tolist())
    eligible = [mid for mid in all_ids if mid in genome_id_set]
    if len(eligible) < 2:
        return np.zeros((0, 0), dtype=np.float32), eligible

    genome_id_idx = {int(mid): i for i, mid in enumerate(genome_movie_ids)}
    indices = [genome_id_idx[mid] for mid in eligible]
    sub_matrix = genome_matrix[indices, :]  # shape (n_eligible, 1128)

    sim = cosine_similarity(sub_matrix).astype(np.float32)  # (n_eligible, n_eligible)
    np.fill_diagonal(sim, 0.0)
    return sim, eligible


# ---------------------------------------------------------------------------
# Audience Jaccard (sparse)
# ---------------------------------------------------------------------------

def _audience_jaccard(
    all_ids: List[int],
    fans: Dict[int, Set[int]],
) -> Tuple[np.ndarray, List[int]]:
    """Return (jaccard_matrix, eligible_ids) based on high-rating user overlap."""
    eligible = [mid for mid in all_ids if len(fans.get(mid, set())) >= MIN_SHARED_FANS]
    if len(eligible) < 2:
        return np.zeros((0, 0), dtype=np.float32), eligible

    all_users = sorted({u for mid in eligible for u in fans[mid]})
    user_idx = {u: i for i, u in enumerate(all_users)}
    movie_idx_map = {mid: i for i, mid in enumerate(eligible)}
    n, m = len(eligible), len(all_users)

    rows, cols = [], []
    for mid in eligible:
        mi = movie_idx_map[mid]
        for uid in fans[mid]:
            rows.append(mi)
            cols.append(user_idx[uid])

    M = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, m)
    )
    intersection = (M @ M.T).toarray()
    counts = np.array(M.sum(axis=1)).flatten()
    union = counts[:, None] + counts[None, :] - intersection

    intersection[intersection < MIN_SHARED_FANS] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, intersection / union, 0.0)
    np.fill_diagonal(jac, 0.0)
    return jac.astype(np.float32), eligible


# ---------------------------------------------------------------------------
# Genre Jaccard (vectorised)
# ---------------------------------------------------------------------------

def _genre_jaccard(
    all_ids: List[int],
    movies: Dict[int, Movie],
) -> Tuple[np.ndarray, List[int]]:
    eligible = [mid for mid in all_ids if movies[mid].genres]
    if len(eligible) < 2:
        return np.zeros((0, 0), dtype=np.float32), eligible

    all_genres = sorted({g for mid in eligible for g in movies[mid].genres})
    genre_idx = {g: i for i, g in enumerate(all_genres)}
    n, m = len(eligible), len(all_genres)

    rows, cols = [], []
    for i, mid in enumerate(eligible):
        for g in movies[mid].genres:
            rows.append(i)
            cols.append(genre_idx[g])

    M = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, m)
    )
    intersection = (M @ M.T).toarray()
    counts = np.array(M.sum(axis=1)).flatten()
    union = counts[:, None] + counts[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, intersection / union, 0.0)
    np.fill_diagonal(jac, 0.0)
    return jac.astype(np.float32), eligible


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _build_graph(
    movies: Dict[int, Movie],
    _fans: Optional[Dict[int, Set[int]]] = None,
    _genome: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> nx.Graph:
    """Build the graph.

    The *_fans* and *_genome* parameters are for unit testing only — pass
    pre-computed data to skip the real data-loading calls.
    """
    all_ids = sorted(movies.keys())
    n = len(all_ids)
    idx_of = {mid: i for i, mid in enumerate(all_ids)}

    print(f"Computing genome cosine similarity for {n} movies …")
    genome_sim, genome_ids = _genome_cosine(all_ids, _genome_override=_genome)
    print(f"  {len(genome_ids):,} movies have genome vectors")

    print("Streaming high ratings for audience matrix …")
    fans = _fans if _fans is not None else load_high_ratings(
        set(all_ids), min_rating=MIN_RATING_FOR_AUDIENCE
    )

    print("Computing audience Jaccard …")
    aud_jac, aud_ids = _audience_jaccard(all_ids, fans)

    print("Computing genre Jaccard …")
    gen_jac, gen_ids = _genre_jaccard(all_ids, movies)

    # ---- Composite weight matrix ----
    weights = np.zeros((n, n), dtype=np.float32)

    if len(genome_ids) >= 2:
        gi = [idx_of[mid] for mid in genome_ids]
        weights[np.ix_(gi, gi)] += genome_sim * _W_GENOME

    if len(aud_ids) >= 2:
        ai = [idx_of[mid] for mid in aud_ids]
        weights[np.ix_(ai, ai)] += aud_jac * _W_AUD

    if len(gen_ids) >= 2:
        gni = [idx_of[mid] for mid in gen_ids]
        weights[np.ix_(gni, gni)] += gen_jac * _W_GEN

    np.fill_diagonal(weights, 0.0)

    # Fast lookups for per-edge metadata
    genome_lookup = {mid: i for i, mid in enumerate(genome_ids)}
    aud_lookup = {mid: i for i, mid in enumerate(aud_ids)}
    gen_lookup = {mid: i for i, mid in enumerate(gen_ids)}

    print("Assembling NetworkX graph …")
    G = nx.Graph()
    for mid, movie in movies.items():
        G.add_node(mid, movie=movie)

    r_arr, c_arr = np.where(weights >= MIN_EDGE_WEIGHT)
    mask = r_arr < c_arr
    r_arr, c_arr = r_arr[mask], c_arr[mask]

    for r, c in zip(r_arr, c_arr):
        m1_id, m2_id = all_ids[r], all_ids[c]
        w = float(weights[r, c])

        gs = (
            float(genome_sim[genome_lookup[m1_id], genome_lookup[m2_id]])
            if m1_id in genome_lookup and m2_id in genome_lookup
            else 0.0
        )
        au = (
            float(aud_jac[aud_lookup[m1_id], aud_lookup[m2_id]])
            if m1_id in aud_lookup and m2_id in aud_lookup
            else 0.0
        )
        gn = (
            float(gen_jac[gen_lookup[m1_id], gen_lookup[m2_id]])
            if m1_id in gen_lookup and m2_id in gen_lookup
            else 0.0
        )

        # Shared user tags for display (not used for weight)
        shared_tags = sorted(
            t for t in (movies[m1_id].tags & movies[m2_id].tags)
            if isinstance(t, str)
        )[:6]

        rel = EdgeRelationship(
            tag_similarity=gs,      # repurposed: genome cosine
            audience_overlap=au,
            genre_similarity=gn,
            shared_tags=shared_tags,
            weight=w,
        )
        G.add_edge(m1_id, m2_id, weight=w, relationship=rel)

    print(
        f"Graph built: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges"
    )
    return G


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_build_graph(
    movies: Dict[int, Movie],
    force_rebuild: bool = False,
) -> nx.Graph:
    """Load graph from cache or build from scratch."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not force_rebuild and GRAPH_CACHE.exists():
        print(f"Loading graph from cache: {GRAPH_CACHE}")
        with open(GRAPH_CACHE, "rb") as fh:
            G = pickle.load(fh)
        for mid, movie in movies.items():
            if mid in G:
                G.nodes[mid]["movie"] = movie
        return G

    G = _build_graph(movies)

    with open(GRAPH_CACHE, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Graph cached to {GRAPH_CACHE}")
    return G


def invalidate_cache() -> None:
    if GRAPH_CACHE.exists():
        GRAPH_CACHE.unlink()
        print("Graph cache cleared.")
