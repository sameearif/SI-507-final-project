"""Tests for src/algorithms.py — search, path finding, centrality, recommendations."""
import pytest
import networkx as nx
from src.models import EdgeRelationship, Movie
from src import algorithms as alg


# ---------------------------------------------------------------------------
# Minimal test graph factory
# ---------------------------------------------------------------------------
#
# Graph layout (5 movies, bidirectional edges):
#
#   Toy Story (1) ── Shrek (2) ── The Matrix (3) ── Interstellar (4)
#                                                          |
#                                                   Inception (5)
#
# Edge weights are given so tests can verify ordering.


def _movie(mid, title, year, genres, rating=4.0, count=100, tags=None):
    return Movie(
        movie_id=mid,
        title=title,
        year=year,
        genres=genres,
        avg_rating=rating,
        rating_count=count,
        tags=set(tags or []),
    )


TOY_STORY = _movie(1, "Toy Story", 1995, ["Animation", "Comedy"], rating=4.0, count=500, tags=["fun", "family"])
SHREK = _movie(2, "Shrek", 2001, ["Animation", "Comedy"], rating=3.8, count=400, tags=["fun", "fairy-tale"])
MATRIX = _movie(3, "The Matrix", 1999, ["Action", "Sci-Fi"], rating=4.5, count=800, tags=["cyberpunk", "reality"])
INTERSTELLAR = _movie(4, "Interstellar", 2014, ["Sci-Fi", "Drama"], rating=4.3, count=700, tags=["space", "reality"])
INCEPTION = _movie(5, "Inception", 2010, ["Sci-Fi", "Thriller"], rating=4.4, count=600, tags=["dreams", "mind"])
ORPHAN = _movie(6, "Obscure Film", 1940, ["Mystery"], rating=2.0, count=5, tags=[])

ALL_MOVIES = {m.movie_id: m for m in [TOY_STORY, SHREK, MATRIX, INTERSTELLAR, INCEPTION, ORPHAN]}


def make_graph() -> nx.Graph:
    G = nx.Graph()
    for mid, movie in ALL_MOVIES.items():
        G.add_node(mid, movie=movie)

    def _rel(ts=0.0, ao=0.0, gs=0.0, tags=None, w=0.3):
        return EdgeRelationship(
            tag_similarity=ts,
            audience_overlap=ao,
            genre_similarity=gs,
            shared_tags=tags or [],
            weight=w,
        )

    G.add_edge(1, 2, weight=0.6, relationship=_rel(ts=0.5, gs=0.8, tags=["fun"], w=0.6))
    G.add_edge(2, 3, weight=0.2, relationship=_rel(ao=0.2, w=0.2))
    G.add_edge(3, 4, weight=0.5, relationship=_rel(ts=0.3, gs=0.5, tags=["reality"], w=0.5))
    G.add_edge(4, 5, weight=0.7, relationship=_rel(ts=0.4, gs=0.6, w=0.7))
    # Node 6 (ORPHAN) has no edges — isolated
    return G


@pytest.fixture
def G():
    return make_graph()


# ---------------------------------------------------------------------------
# search_movies
# ---------------------------------------------------------------------------


def test_search_exact_title():
    results = alg.search_movies(ALL_MOVIES, "Inception")
    assert any(m.movie_id == 5 for m in results)


def test_search_case_insensitive():
    results = alg.search_movies(ALL_MOVIES, "inception")
    assert any(m.movie_id == 5 for m in results)


def test_search_partial_match():
    results = alg.search_movies(ALL_MOVIES, "matrix")
    assert any(m.movie_id == 3 for m in results)


def test_search_empty_query_returns_empty():
    results = alg.search_movies(ALL_MOVIES, "")
    assert results == []


def test_search_no_match_returns_empty():
    results = alg.search_movies(ALL_MOVIES, "xyzzy_nonexistent")
    assert results == []


def test_search_respects_max_results():
    results = alg.search_movies(ALL_MOVIES, "e", max_results=2)
    assert len(results) <= 2


def test_search_sorted_by_rating_count():
    results = alg.search_movies(ALL_MOVIES, "s")  # matches Shrek, The Matrix, etc.
    counts = [m.rating_count for m in results]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# filter_movies
# ---------------------------------------------------------------------------


def test_filter_by_genre():
    results = alg.filter_movies(ALL_MOVIES, genres=["Animation"])
    ids = {m.movie_id for m in results}
    assert 1 in ids and 2 in ids
    assert 3 not in ids


def test_filter_by_multiple_genres_all_must_match():
    results = alg.filter_movies(ALL_MOVIES, genres=["Sci-Fi", "Drama"])
    ids = {m.movie_id for m in results}
    assert ids == {4}  # Only Interstellar has both


def test_filter_by_year_range():
    results = alg.filter_movies(ALL_MOVIES, year_range=(2010, 2014))
    ids = {m.movie_id for m in results}
    assert 4 in ids and 5 in ids
    assert 1 not in ids


def test_filter_by_min_rating():
    results = alg.filter_movies(ALL_MOVIES, min_avg_rating=4.3)
    for m in results:
        assert m.avg_rating >= 4.3


def test_filter_by_tag():
    results = alg.filter_movies(ALL_MOVIES, tag="family")
    assert any(m.movie_id == 1 for m in results)
    assert all("family" in m.tags for m in results)


def test_filter_no_criteria_returns_all():
    results = alg.filter_movies(ALL_MOVIES)
    assert len(results) == len(ALL_MOVIES)


def test_filter_impossible_criteria_returns_empty():
    results = alg.filter_movies(ALL_MOVIES, genres=["Animation", "Sci-Fi"])
    assert results == []


# ---------------------------------------------------------------------------
# get_neighborhood
# ---------------------------------------------------------------------------


def test_neighborhood_depth_1(G):
    sub = alg.get_neighborhood(G, movie_id=3, depth=1)
    # Node 3 connects to 2 and 4
    assert 3 in sub
    assert 2 in sub
    assert 4 in sub
    assert 1 not in sub  # two hops away


def test_neighborhood_depth_2(G):
    sub = alg.get_neighborhood(G, movie_id=3, depth=2)
    # Depth 2 from node 3: 2,4 (depth 1), 1,5 (depth 2)
    assert 1 in sub
    assert 5 in sub


def test_neighborhood_isolated_node(G):
    sub = alg.get_neighborhood(G, movie_id=6, depth=2)
    # Orphan has no neighbours
    assert 6 in sub
    assert sub.number_of_nodes() == 1


def test_neighborhood_missing_node(G):
    sub = alg.get_neighborhood(G, movie_id=999, depth=1)
    assert sub.number_of_nodes() == 0


def test_neighborhood_max_nodes_respected(G):
    sub = alg.get_neighborhood(G, movie_id=3, depth=2, max_nodes=3)
    assert sub.number_of_nodes() <= 3


# ---------------------------------------------------------------------------
# find_path
# ---------------------------------------------------------------------------


def test_find_path_direct(G):
    path = alg.find_path(G, 1, 2)
    assert path == [1, 2]


def test_find_path_multi_hop(G):
    path = alg.find_path(G, 1, 4)
    assert path is not None
    assert path[0] == 1
    assert path[-1] == 4
    # Verify every step is an actual edge
    for a, b in zip(path[:-1], path[1:]):
        assert G.has_edge(a, b)


def test_find_path_no_path_to_isolated(G):
    path = alg.find_path(G, 1, 6)
    assert path is None


def test_find_path_missing_node(G):
    path = alg.find_path(G, 1, 999)
    assert path is None


def test_find_path_same_node(G):
    path = alg.find_path(G, 3, 3)
    assert path == [3]


# ---------------------------------------------------------------------------
# path_relationships
# ---------------------------------------------------------------------------


def test_path_relationships_length(G):
    path = alg.find_path(G, 1, 4)
    rels = alg.path_relationships(G, path)
    assert len(rels) == len(path) - 1


def test_path_relationships_types(G):
    path = [1, 2, 3]
    rels = alg.path_relationships(G, path)
    for m_a, m_b, rel in rels:
        assert isinstance(m_a, Movie)
        assert isinstance(m_b, Movie)
        assert isinstance(rel, EdgeRelationship)


def test_path_relationships_correct_movies(G):
    path = [1, 2]
    rels = alg.path_relationships(G, path)
    assert rels[0][0].movie_id == 1
    assert rels[0][1].movie_id == 2


def test_path_relationships_empty_path(G):
    assert alg.path_relationships(G, []) == []


def test_path_relationships_single_node(G):
    assert alg.path_relationships(G, [1]) == []


# ---------------------------------------------------------------------------
# top_by_centrality
# ---------------------------------------------------------------------------


def test_top_by_degree_returns_list(G):
    ranked = alg.top_by_centrality(G, metric="degree")
    assert isinstance(ranked, list)
    assert all(isinstance(m, Movie) for m, _ in ranked)


def test_top_by_degree_sorted_descending(G):
    ranked = alg.top_by_centrality(G, metric="degree", n=10)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_top_by_centrality_respects_n(G):
    ranked = alg.top_by_centrality(G, metric="degree", n=2)
    assert len(ranked) <= 2


def test_top_by_centrality_genre_filter(G):
    ranked = alg.top_by_centrality(G, metric="degree", genre_filter="Animation")
    for movie, _ in ranked:
        assert "Animation" in movie.genres


def test_top_by_centrality_year_filter(G):
    ranked = alg.top_by_centrality(G, metric="degree", year_range=(2009, 2015))
    for movie, _ in ranked:
        assert movie.year is not None
        assert 2009 <= movie.year <= 2015


def test_top_by_centrality_unknown_metric_raises(G):
    with pytest.raises(ValueError, match="Unknown metric"):
        alg.top_by_centrality(G, metric="notreal")


def test_top_by_pagerank(G):
    ranked = alg.top_by_centrality(G, metric="pagerank")
    assert len(ranked) > 0


def test_top_by_betweenness(G):
    ranked = alg.top_by_centrality(G, metric="betweenness")
    assert len(ranked) > 0


# ---------------------------------------------------------------------------
# get_recommendations
# ---------------------------------------------------------------------------


def test_recommendations_basic(G):
    recs = alg.get_recommendations(G, movie_id=3, n=5)
    assert isinstance(recs, list)
    for movie, score, reason in recs:
        assert isinstance(movie, Movie)
        assert isinstance(score, float)
        assert isinstance(reason, str)


def test_recommendations_excludes_seed(G):
    recs = alg.get_recommendations(G, movie_id=3, n=10)
    ids = {m.movie_id for m, _, _ in recs}
    assert 3 not in ids


def test_recommendations_sorted_by_score(G):
    recs = alg.get_recommendations(G, movie_id=3, n=10)
    scores = [s for _, s, _ in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommendations_for_isolated_node_returns_empty(G):
    recs = alg.get_recommendations(G, movie_id=6)
    assert recs == []


def test_recommendations_missing_node(G):
    recs = alg.get_recommendations(G, movie_id=999)
    assert recs == []


def test_recommendations_respects_n(G):
    recs = alg.get_recommendations(G, movie_id=1, n=2)
    assert len(recs) <= 2


# ---------------------------------------------------------------------------
# graph_stats
# ---------------------------------------------------------------------------


def test_graph_stats_keys(G):
    stats = alg.graph_stats(G)
    for key in ("nodes", "edges", "density", "connected_components",
                "largest_component_size", "average_degree"):
        assert key in stats


def test_graph_stats_node_count(G):
    stats = alg.graph_stats(G)
    assert stats["nodes"] == G.number_of_nodes()


def test_graph_stats_edge_count(G):
    stats = alg.graph_stats(G)
    assert stats["edges"] == G.number_of_edges()


def test_graph_stats_two_components(G):
    # Node 6 is isolated → 2 components (1-2-3-4-5 and 6)
    stats = alg.graph_stats(G)
    assert stats["connected_components"] == 2


def test_graph_stats_largest_component(G):
    stats = alg.graph_stats(G)
    assert stats["largest_component_size"] == 5
