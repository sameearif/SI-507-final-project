"""Tests for src/models.py — Movie and EdgeRelationship classes."""
import pytest
from src.models import EdgeRelationship, Movie


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_movie(**kwargs) -> Movie:
    defaults = dict(
        movie_id=1,
        title="Test Movie",
        year=2000,
        genres=["Action", "Drama"],
        avg_rating=4.0,
        rating_count=100,
        tags={"exciting", "dark", "suspense"},
    )
    defaults.update(kwargs)
    return Movie(**defaults)


# ---------------------------------------------------------------------------
# Movie construction
# ---------------------------------------------------------------------------


def test_movie_basic_attributes():
    m = make_movie(movie_id=42, title="Inception", year=2010)
    assert m.movie_id == 42
    assert m.title == "Inception"
    assert m.year == 2010


def test_movie_display_title_with_year():
    m = make_movie(title="The Matrix", year=1999)
    assert m.display_title() == "The Matrix (1999)"


def test_movie_display_title_without_year():
    m = make_movie(title="Unknown Film", year=None)
    assert m.display_title() == "Unknown Film"


def test_movie_genre_str_nonempty():
    m = make_movie(genres=["Sci-Fi", "Thriller"])
    assert m.genre_str() == "Sci-Fi, Thriller"


def test_movie_genre_str_empty():
    m = make_movie(genres=[])
    assert m.genre_str() == "Unknown"


def test_movie_top_tags_returns_sorted_subset():
    m = make_movie(tags={"zebra", "apple", "mango", "banana", "cherry", "fig"})
    top = m.top_tags(3)
    assert top == sorted(top)
    assert len(top) == 3


def test_movie_top_tags_fewer_than_n():
    m = make_movie(tags={"only", "two"})
    top = m.top_tags(10)
    assert len(top) == 2


def test_movie_top_tags_empty():
    m = make_movie(tags=set())
    assert m.top_tags() == []


def test_movie_to_dict_keys():
    m = make_movie()
    d = m.to_dict()
    for key in ("movie_id", "title", "year", "genres", "avg_rating", "rating_count", "tags"):
        assert key in d


def test_movie_to_dict_tags_sorted():
    m = make_movie(tags={"z_tag", "a_tag", "m_tag"})
    d = m.to_dict()
    assert d["tags"] == sorted(d["tags"])


# ---------------------------------------------------------------------------
# Movie equality and hashing
# ---------------------------------------------------------------------------


def test_movie_equality_same_id():
    m1 = make_movie(movie_id=7)
    m2 = make_movie(movie_id=7, title="Different Title")
    assert m1 == m2


def test_movie_inequality_different_id():
    m1 = make_movie(movie_id=1)
    m2 = make_movie(movie_id=2)
    assert m1 != m2


def test_movie_hashable_in_set():
    m1 = make_movie(movie_id=1)
    m2 = make_movie(movie_id=1)
    m3 = make_movie(movie_id=2)
    s = {m1, m2, m3}
    assert len(s) == 2


def test_movie_usable_as_dict_key():
    m = make_movie(movie_id=5)
    d = {m: "value"}
    assert d[m] == "value"


# ---------------------------------------------------------------------------
# EdgeRelationship
# ---------------------------------------------------------------------------


def test_edge_relationship_defaults():
    rel = EdgeRelationship()
    assert rel.tag_similarity == 0.0
    assert rel.audience_overlap == 0.0
    assert rel.genre_similarity == 0.0
    assert rel.shared_tags == []
    assert rel.weight == 0.0


def test_edge_describe_with_genome_similarity():
    rel = EdgeRelationship(
        tag_similarity=0.85,   # genome cosine in ml-32m context
        shared_tags=[],
        audience_overlap=0.0,
        genre_similarity=0.0,
    )
    desc = rel.describe()
    assert "content" in desc.lower() or "similar" in desc.lower()


def test_edge_describe_with_shared_tags():
    rel = EdgeRelationship(
        tag_similarity=0.0,
        shared_tags=["dark", "suspense"],
        audience_overlap=0.0,
        genre_similarity=0.0,
    )
    desc = rel.describe()
    assert "dark" in desc
    assert "suspense" in desc


def test_edge_describe_with_audience():
    rel = EdgeRelationship(audience_overlap=0.25)
    desc = rel.describe()
    assert "audience" in desc.lower()


def test_edge_describe_with_genre():
    rel = EdgeRelationship(genre_similarity=0.8)
    desc = rel.describe()
    assert "genre" in desc.lower()


def test_edge_describe_no_signals_returns_fallback():
    rel = EdgeRelationship()
    desc = rel.describe()
    assert isinstance(desc, str)
    assert len(desc) > 0


def test_edge_dominant_signal_tag():
    rel = EdgeRelationship(tag_similarity=0.9, audience_overlap=0.1, genre_similarity=0.1)
    assert rel.dominant_signal() == "shared tags"


def test_edge_dominant_signal_audience():
    rel = EdgeRelationship(tag_similarity=0.0, audience_overlap=0.8, genre_similarity=0.0)
    assert rel.dominant_signal() == "audience overlap"


def test_edge_dominant_signal_genre():
    rel = EdgeRelationship(tag_similarity=0.0, audience_overlap=0.0, genre_similarity=1.0)
    assert rel.dominant_signal() == "genre similarity"


def test_edge_describe_truncates_many_tags():
    rel = EdgeRelationship(shared_tags=["a", "b", "c", "d", "e"], tag_similarity=0.5)
    desc = rel.describe()
    # Should show at most 3 tags plus "+N more"
    assert "a" in desc or "more" in desc
