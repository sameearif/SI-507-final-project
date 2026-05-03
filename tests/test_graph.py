"""Tests for src/graph_builder.py similarity matrix helpers and graph assembly.

These tests use tiny synthetic data so they run instantly without downloading
any datasets.  The genome, audience, and genre similarity functions are each
tested independently.
"""
import numpy as np
import networkx as nx
import pandas as pd
import pytest

from src.models import Movie, EdgeRelationship
from src.graph_builder import (
    _genome_cosine,
    _audience_jaccard,
    _genre_jaccard,
    _build_graph,
    MIN_EDGE_WEIGHT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _m(mid, genres, tags=None, rating_count=200):
    return Movie(
        movie_id=mid,
        title=f"Movie {mid}",
        year=2000,
        genres=genres,
        rating_count=rating_count,
        tags=set(tags or []),
    )


def _ratings_df(entries):
    return pd.DataFrame(entries, columns=["userId", "movieId", "rating"])


def _synthetic_genome(movie_ids, n_tags=8):
    """Return (id_array, matrix) with random unit-norm vectors."""
    rng = np.random.default_rng(42)
    ids = np.array(movie_ids, dtype=np.int32)
    mat = rng.random((len(ids), n_tags)).astype(np.float32)
    # Normalise rows so cosine similarity is well-defined
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat /= np.maximum(norms, 1e-8)
    return ids, mat


# ---------------------------------------------------------------------------
# _genome_cosine
# ---------------------------------------------------------------------------


def test_genome_cosine_identical_vectors():
    """Two movies with the same genome vector should have cosine = 1."""
    ids = np.array([1, 2], dtype=np.int32)
    vec = np.array([[1, 0, 0, 0]], dtype=np.float32)
    mat = np.vstack([vec, vec])  # identical rows
    sim, eligible = _genome_cosine([1, 2], _genome_override=(ids, mat))
    idx1, idx2 = eligible.index(1), eligible.index(2)
    assert sim[idx1, idx2] == pytest.approx(1.0, abs=1e-5)


def test_genome_cosine_orthogonal_vectors():
    """Orthogonal genome vectors should have cosine = 0."""
    ids = np.array([1, 2], dtype=np.int32)
    mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sim, eligible = _genome_cosine([1, 2], _genome_override=(ids, mat))
    idx1, idx2 = eligible.index(1), eligible.index(2)
    assert sim[idx1, idx2] == pytest.approx(0.0, abs=1e-5)


def test_genome_cosine_diagonal_zero():
    ids, mat = _synthetic_genome([10, 20, 30])
    sim, eligible = _genome_cosine([10, 20, 30], _genome_override=(ids, mat))
    for i in range(len(eligible)):
        assert sim[i, i] == pytest.approx(0.0, abs=1e-5)


def test_genome_cosine_symmetric():
    ids, mat = _synthetic_genome([1, 2])
    sim, eligible = _genome_cosine([1, 2], _genome_override=(ids, mat))
    idx1, idx2 = eligible.index(1), eligible.index(2)
    assert sim[idx1, idx2] == pytest.approx(sim[idx2, idx1], abs=1e-5)


def test_genome_cosine_returns_only_eligible_ids():
    """Only movies present in the genome data should be in eligible."""
    ids = np.array([10, 20], dtype=np.int32)
    mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    # Pass [10, 20, 30, 40] but genome only has 10 and 20
    sim, eligible = _genome_cosine([10, 20, 30, 40], _genome_override=(ids, mat))
    assert set(eligible) == {10, 20}
    assert 30 not in eligible
    assert 40 not in eligible


def test_genome_cosine_no_genome_returns_empty():
    sim, eligible = _genome_cosine([1, 2, 3], _genome_override=None)
    assert len(eligible) == 0
    assert sim.shape == (0, 0)


def test_genome_cosine_values_in_range():
    ids, mat = _synthetic_genome([1, 2, 3, 4])
    sim, eligible = _genome_cosine([1, 2, 3, 4], _genome_override=(ids, mat))
    assert np.all(sim >= 0.0)
    assert np.all(sim <= 1.0 + 1e-5)


# ---------------------------------------------------------------------------
# _audience_jaccard
# ---------------------------------------------------------------------------


def test_audience_jaccard_identical_fans():
    fans = {10: {1, 2, 3, 4, 5, 6}, 20: {1, 2, 3, 4, 5, 6}}
    jac, ids = _audience_jaccard([10, 20], fans)
    assert len(ids) >= 2
    idx10, idx20 = ids.index(10), ids.index(20)
    assert jac[idx10, idx20] == pytest.approx(1.0)


def test_audience_jaccard_no_shared_fans():
    fans = {10: {1, 2, 3, 4, 5}, 20: {6, 7, 8, 9, 10}}
    jac, ids = _audience_jaccard([10, 20], fans)
    if len(ids) >= 2:
        idx10, idx20 = ids.index(10), ids.index(20)
        assert jac[idx10, idx20] == pytest.approx(0.0)


def test_audience_jaccard_partial_overlap():
    # fans 1-5 for both movies, fans 6-10 unique to movie 20
    # intersection=5, union=10 → Jaccard=0.5
    fans = {10: {1, 2, 3, 4, 5}, 20: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
    jac, ids = _audience_jaccard([10, 20], fans)
    if len(ids) >= 2:
        idx10, idx20 = ids.index(10), ids.index(20)
        assert jac[idx10, idx20] == pytest.approx(0.5, abs=0.01)


def test_audience_jaccard_diagonal_zero():
    fans = {1: {1, 2, 3, 4, 5, 6}, 2: {1, 2, 3, 4, 5, 6}}
    jac, ids = _audience_jaccard([1, 2], fans)
    for i in range(len(ids)):
        assert jac[i, i] == pytest.approx(0.0)


def test_audience_jaccard_symmetric():
    fans = {1: {1, 2, 3, 4, 5}, 2: {3, 4, 5, 6, 7, 8}}
    jac, ids = _audience_jaccard([1, 2], fans)
    if len(ids) >= 2:
        idx1, idx2 = ids.index(1), ids.index(2)
        assert jac[idx1, idx2] == pytest.approx(jac[idx2, idx1])


def test_audience_jaccard_excludes_insufficient_fans():
    # Movie 99 has only 2 fans → below MIN_SHARED_FANS → excluded
    fans = {1: {1, 2, 3, 4, 5}, 99: {6, 7}}
    jac, ids = _audience_jaccard([1, 99], fans)
    assert 99 not in ids


def test_audience_jaccard_missing_movie_excluded():
    fans = {1: {1, 2, 3, 4, 5}}
    jac, ids = _audience_jaccard([1, 2], fans)  # movie 2 not in fans
    assert 2 not in ids


# ---------------------------------------------------------------------------
# _genre_jaccard
# ---------------------------------------------------------------------------


def test_genre_jaccard_same_genres():
    movies = {1: _m(1, ["Drama", "Romance"]), 2: _m(2, ["Drama", "Romance"])}
    jac, ids = _genre_jaccard([1, 2], movies)
    idx1, idx2 = ids.index(1), ids.index(2)
    assert jac[idx1, idx2] == pytest.approx(1.0)


def test_genre_jaccard_disjoint_genres():
    movies = {1: _m(1, ["Comedy"]), 2: _m(2, ["Horror"])}
    jac, ids = _genre_jaccard([1, 2], movies)
    idx1, idx2 = ids.index(1), ids.index(2)
    assert jac[idx1, idx2] == pytest.approx(0.0)


def test_genre_jaccard_partial_overlap():
    # {Drama, Action} vs {Drama, Comedy} → intersection=1, union=3 → 0.333
    movies = {1: _m(1, ["Drama", "Action"]), 2: _m(2, ["Drama", "Comedy"])}
    jac, ids = _genre_jaccard([1, 2], movies)
    idx1, idx2 = ids.index(1), ids.index(2)
    assert jac[idx1, idx2] == pytest.approx(1 / 3, abs=0.01)


def test_genre_jaccard_diagonal_zero():
    movies = {1: _m(1, ["Drama"]), 2: _m(2, ["Drama"])}
    jac, ids = _genre_jaccard([1, 2], movies)
    for i in range(len(ids)):
        assert jac[i, i] == pytest.approx(0.0)


def test_genre_jaccard_excludes_movies_with_no_genres():
    movies = {1: _m(1, []), 2: _m(2, ["Drama"]), 3: _m(3, ["Drama"])}
    jac, ids = _genre_jaccard([1, 2, 3], movies)
    assert 1 not in ids


def test_genre_jaccard_symmetric():
    movies = {1: _m(1, ["Action", "Drama"]), 2: _m(2, ["Drama", "Romance"])}
    jac, ids = _genre_jaccard([1, 2], movies)
    idx1, idx2 = ids.index(1), ids.index(2)
    assert jac[idx1, idx2] == pytest.approx(jac[idx2, idx1])


# ---------------------------------------------------------------------------
# _build_graph integration (lightweight — uses injected data, no downloads)
# ---------------------------------------------------------------------------


def _make_movies(n=5):
    return {
        i: _m(i, ["Drama", "Action"], ["epic", "dark"], rating_count=200)
        for i in range(1, n + 1)
    }


def _make_fans(movie_ids, n_users=20):
    """Each movie gets n_users fans, with 50% overlap between movies."""
    fans = {}
    for mid in movie_ids:
        fans[mid] = set(range(mid * 5, mid * 5 + n_users))
    # Force some overlap between all pairs so edges form
    shared = set(range(1000, 1000 + n_users))
    for mid in movie_ids:
        fans[mid].update(shared)
    return fans


def test_build_graph_correct_node_count():
    movies = _make_movies(4)
    fans = _make_fans(list(movies))
    G = _build_graph(movies, _fans=fans, _genome=None)
    assert G.number_of_nodes() == 4


def test_build_graph_all_movies_are_nodes():
    movies = _make_movies(3)
    fans = _make_fans(list(movies))
    G = _build_graph(movies, _fans=fans, _genome=None)
    for mid in movies:
        assert mid in G


def test_build_graph_edges_have_weight():
    movies = _make_movies(3)
    fans = _make_fans(list(movies))
    G = _build_graph(movies, _fans=fans, _genome=None)
    for _, _, data in G.edges(data=True):
        assert "weight" in data
        assert data["weight"] >= MIN_EDGE_WEIGHT


def test_build_graph_edges_have_relationship():
    movies = _make_movies(3)
    fans = _make_fans(list(movies))
    G = _build_graph(movies, _fans=fans, _genome=None)
    for u, v, data in G.edges(data=True):
        assert isinstance(data.get("relationship"), EdgeRelationship)


def test_build_graph_no_self_loops():
    movies = _make_movies(4)
    fans = _make_fans(list(movies))
    G = _build_graph(movies, _fans=fans, _genome=None)
    assert not any(u == v for u, v in G.edges())


def test_build_graph_with_genome_creates_edges():
    movies = _make_movies(3)
    fans = _make_fans(list(movies))
    ids, mat = _synthetic_genome(list(movies.keys()), n_tags=10)
    # Make first two movies very similar
    mat[0] = mat[1]
    G = _build_graph(movies, _fans=fans, _genome=(ids, mat))
    # Movies 1 and 2 (first two) should definitely be connected
    assert G.has_edge(1, 2)


def test_build_graph_genome_boosts_similar_movies():
    """Genome-similar movies should have higher edge weight."""
    movies = _make_movies(3)
    fans = _make_fans(list(movies))
    ids = np.array([1, 2, 3], dtype=np.int32)
    # Movie 1 and 2 identical; movie 3 orthogonal to both
    mat = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    G = _build_graph(movies, _fans=fans, _genome=(ids, mat))
    if G.has_edge(1, 2) and G.has_edge(1, 3):
        w12 = G[1][2]["weight"]
        w13 = G[1][3]["weight"]
        assert w12 > w13


def test_build_graph_node_movie_attribute():
    movies = _make_movies(2)
    fans = _make_fans(list(movies))
    G = _build_graph(movies, _fans=fans, _genome=None)
    for mid, movie in movies.items():
        assert G.nodes[mid]["movie"] is movie
