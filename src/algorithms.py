"""Graph algorithms for the Movie Discovery Graph.

All functions operate on a NetworkX Graph whose nodes are movie IDs (int) and
whose node attribute 'movie' holds a Movie object.  Edge attribute 'weight'
is the composite similarity score; edge attribute 'relationship' is an
EdgeRelationship instance.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx

from src.models import EdgeRelationship, Movie


# ---------------------------------------------------------------------------
# Search & filter
# ---------------------------------------------------------------------------

def search_movies(
    movies: Dict[int, Movie],
    query: str,
    max_results: int = 20,
) -> List[Movie]:
    """Case-insensitive substring search over movie titles.

    Returns up to *max_results* movies sorted by rating_count descending
    so the most-recognised matches appear first.
    """
    q = query.strip().lower()
    if not q:
        return []
    hits = [
        m for m in movies.values()
        if q in m.title.lower() or q in m.display_title().lower()
    ]
    hits.sort(key=lambda m: m.rating_count, reverse=True)
    return hits[:max_results]


def filter_movies(
    movies: Dict[int, Movie],
    genres: Optional[List[str]] = None,
    year_range: Optional[Tuple[int, int]] = None,
    min_avg_rating: float = 0.0,
    tag: Optional[str] = None,
) -> List[Movie]:
    """Filter the movie catalogue by one or more criteria.

    Parameters
    ----------
    genres:
        If given, only movies that contain *all* of the listed genres pass.
    year_range:
        ``(lo, hi)`` inclusive year filter.
    min_avg_rating:
        MovieLens mean rating floor.
    tag:
        Substring match against user tags (case-insensitive).
    """
    results = list(movies.values())

    if genres:
        genre_set = {g.lower() for g in genres}
        results = [
            m for m in results
            if genre_set.issubset({g.lower() for g in m.genres})
        ]

    if year_range:
        lo, hi = year_range
        results = [
            m for m in results
            if m.year is not None and lo <= m.year <= hi
        ]

    if min_avg_rating > 0:
        results = [m for m in results if m.avg_rating >= min_avg_rating]

    if tag:
        tag_q = tag.strip().lower()
        results = [m for m in results if any(tag_q in t for t in m.tags)]

    results.sort(key=lambda m: m.rating_count, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------

def get_neighborhood(
    G: nx.Graph,
    movie_id: int,
    depth: int = 1,
    max_nodes: int = 50,
) -> nx.Graph:
    """Return the ego-graph of *movie_id* up to *depth* hops.

    Caps at *max_nodes* neighbours (sorted by edge weight) to keep
    visualisations manageable.
    """
    if movie_id not in G:
        return nx.Graph()

    # BFS up to depth
    seen = {movie_id}
    frontier = {movie_id}
    for _ in range(depth):
        if len(seen) >= max_nodes:
            break
        next_frontier: set = set()
        for node in frontier:
            neighbours = list(G.neighbors(node))
            # Sort by weight so we take the strongest connections first
            neighbours.sort(
                key=lambda nb: G[node][nb].get("weight", 0.0), reverse=True
            )
            for nb in neighbours:
                if nb not in seen:
                    next_frontier.add(nb)
                    seen.add(nb)
                    if len(seen) >= max_nodes:
                        break
            if len(seen) >= max_nodes:
                break
        frontier = next_frontier
        if not frontier:
            break

    return G.subgraph(seen).copy()


# ---------------------------------------------------------------------------
# Shortest path
# ---------------------------------------------------------------------------

def find_path(
    G: nx.Graph,
    src_id: int,
    dst_id: int,
) -> Optional[List[int]]:
    """Return the shortest (unweighted hop) path between two movies.

    Returns None if either node is absent or the nodes are disconnected.
    We use hop count rather than inverse-weight so the path length is
    intuitive to end users.
    """
    if src_id not in G or dst_id not in G:
        return None
    try:
        return nx.shortest_path(G, src_id, dst_id)
    except nx.NetworkXNoPath:
        return None


def path_relationships(
    G: nx.Graph, path: List[int]
) -> List[Tuple[Movie, Movie, EdgeRelationship]]:
    """Extract (movie_a, movie_b, relationship) for each hop in *path*."""
    result = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        m_a: Movie = G.nodes[a]["movie"]
        m_b: Movie = G.nodes[b]["movie"]
        rel: EdgeRelationship = G[a][b].get("relationship", EdgeRelationship())
        result.append((m_a, m_b, rel))
    return result


# ---------------------------------------------------------------------------
# Centrality & rankings
# ---------------------------------------------------------------------------

def _degree_centrality(G: nx.Graph) -> Dict[int, float]:
    return nx.degree_centrality(G)


def _betweenness_centrality(G: nx.Graph) -> Dict[int, float]:
    # Use k-sample approximation for speed on large graphs
    k = min(200, G.number_of_nodes())
    return nx.betweenness_centrality(G, k=k, weight="weight", normalized=True, seed=42)


def _pagerank(G: nx.Graph) -> Dict[int, float]:
    return nx.pagerank(G, weight="weight")


def _closeness_centrality(G: nx.Graph) -> Dict[int, float]:
    return nx.closeness_centrality(G, distance=None)


_CENTRALITY_FNS = {
    "degree": _degree_centrality,
    "betweenness": _betweenness_centrality,
    "pagerank": _pagerank,
    "closeness": _closeness_centrality,
}

CENTRALITY_LABELS = {
    "degree": "Degree Centrality",
    "betweenness": "Betweenness Centrality",
    "pagerank": "PageRank",
    "closeness": "Closeness Centrality",
}


def top_by_centrality(
    G: nx.Graph,
    metric: str = "degree",
    n: int = 20,
    genre_filter: Optional[str] = None,
    year_range: Optional[Tuple[int, int]] = None,
) -> List[Tuple[Movie, float]]:
    """Return the top-n movies ranked by *metric*.

    Parameters
    ----------
    metric:
        One of 'degree', 'betweenness', 'pagerank', 'closeness'.
    genre_filter:
        If set, restrict to movies containing this genre.
    year_range:
        ``(lo, hi)`` year filter applied before ranking.
    """
    if metric not in _CENTRALITY_FNS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(_CENTRALITY_FNS)}")

    scores: Dict[int, float] = _CENTRALITY_FNS[metric](G)

    results: List[Tuple[Movie, float]] = []
    for mid, score in scores.items():
        movie: Movie = G.nodes[mid]["movie"]

        if genre_filter and genre_filter.lower() not in {g.lower() for g in movie.genres}:
            continue

        if year_range:
            lo, hi = year_range
            if movie.year is None or not (lo <= movie.year <= hi):
                continue

        results.append((movie, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n]


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def get_recommendations(
    G: nx.Graph,
    movie_id: int,
    n: int = 10,
    exclude_genres: Optional[List[str]] = None,
) -> List[Tuple[Movie, float, str]]:
    """Graph-proximity recommendations for *movie_id*.

    Strategy:
    1. Direct neighbours ranked by edge weight (depth-1).
    2. If fewer than n found, include depth-2 neighbours ranked by max
       path weight to the seed movie.

    Returns a list of (movie, score, reason) triples where *reason* is a
    short human-readable string.
    """
    if movie_id not in G:
        return []

    exclude_set = {g.lower() for g in (exclude_genres or [])}

    seen = {movie_id}
    results: List[Tuple[Movie, float, str]] = []

    # Depth-1: direct neighbours
    for nb in G.neighbors(movie_id):
        movie: Movie = G.nodes[nb]["movie"]
        if exclude_set and exclude_set.issubset({g.lower() for g in movie.genres}):
            continue
        weight = G[movie_id][nb].get("weight", 0.0)
        rel: EdgeRelationship = G[movie_id][nb].get("relationship", EdgeRelationship())
        reason = rel.describe()
        results.append((movie, weight, reason))
        seen.add(nb)

    results.sort(key=lambda x: x[1], reverse=True)

    # Depth-2: fill up to n if needed
    if len(results) < n:
        for nb in list(seen - {movie_id}):
            for nb2 in G.neighbors(nb):
                if nb2 in seen:
                    continue
                movie2: Movie = G.nodes[nb2]["movie"]
                if exclude_set and exclude_set.issubset({g.lower() for g in movie2.genres}):
                    continue
                # Score = average of two edge weights
                w1 = G[movie_id][nb].get("weight", 0.0)
                w2 = G[nb][nb2].get("weight", 0.0)
                score = (w1 + w2) / 2.0
                via_movie: Movie = G.nodes[nb]["movie"]
                reason = f"via {via_movie.display_title()}"
                results.append((movie2, score, reason))
                seen.add(nb2)

        results.sort(key=lambda x: x[1], reverse=True)

    return results[:n]


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

def graph_stats(G: nx.Graph) -> Dict[str, object]:
    """Return a summary dict of basic graph statistics."""
    components = list(nx.connected_components(G))
    largest = max(len(c) for c in components) if components else 0
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 6),
        "connected_components": len(components),
        "largest_component_size": largest,
        "average_degree": round(
            sum(d for _, d in G.degree()) / max(G.number_of_nodes(), 1), 2
        ),
    }
