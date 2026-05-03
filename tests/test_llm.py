"""Tests for src/llm.py Ollama request shaping."""
from types import SimpleNamespace

from src import llm
from src.models import EdgeRelationship, Movie


def _path_data():
    movie_a = Movie(
        movie_id=1,
        title="Movie A",
        year=2000,
        genres=["Drama"],
        tags={"warm"},
    )
    movie_b = Movie(
        movie_id=2,
        title="Movie B",
        year=2001,
        genres=["Drama"],
        tags={"warm"},
    )
    rel = EdgeRelationship(shared_tags=["warm"], tag_similarity=0.8, weight=0.8)
    return [(movie_a, movie_b, rel)]


def test_generate_story_disables_reasoning_for_qwen35(monkeypatch):
    calls = []

    class DummyClient:
        def __init__(self, host):
            self.host = host

        def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(response="Story")

    monkeypatch.setattr(llm, "_OLLAMA_PKG_AVAILABLE", True)
    monkeypatch.setattr(llm, "_ollama_lib", SimpleNamespace(Client=DummyClient))

    story = llm.generate_story(_path_data(), model="qwen3.5:4b-q4_K_M")

    assert story == "Story"
    assert calls[0]["think"] is False
    assert calls[0]["stream"] is False


def test_stream_story_disables_reasoning_for_qwen35(monkeypatch):
    calls = []

    class DummyClient:
        def __init__(self, host):
            self.host = host

        def generate(self, **kwargs):
            calls.append(kwargs)
            return [
                SimpleNamespace(response="Part 1"),
                SimpleNamespace(response="Part 2"),
            ]

    monkeypatch.setattr(llm, "_OLLAMA_PKG_AVAILABLE", True)
    monkeypatch.setattr(llm, "_ollama_lib", SimpleNamespace(Client=DummyClient))

    chunks = list(llm.stream_story(_path_data(), model="qwen3.5:4b"))

    assert chunks == ["Part 1", "Part 2"]
    assert calls[0]["think"] is False
    assert calls[0]["stream"] is True


def test_generate_story_leaves_other_models_unchanged(monkeypatch):
    calls = []

    class DummyClient:
        def __init__(self, host):
            self.host = host

        def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(response="Story")

    monkeypatch.setattr(llm, "_OLLAMA_PKG_AVAILABLE", True)
    monkeypatch.setattr(llm, "_ollama_lib", SimpleNamespace(Client=DummyClient))

    llm.generate_story(_path_data(), model="mistral:7b")

    assert "think" not in calls[0]
