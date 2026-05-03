"""Local LLM connection story generation via Ollama.

Three modes
-----------
- "off"   : LLM is disabled.  A rule-based fallback is always returned.
- "ollama": Uses a locally running Ollama server (default: http://localhost:11434).
            Supports any model available on the machine, with int4/int8 variants
            selected via the quantisation suffix in the model tag.

Quantisation guide (Ollama tag suffixes)
-----------------------------------------
Default  : ``qwen3.5:4b``              (Ollama picks, usually q4_K_M)
int4     : ``qwen3.5:4b-q4_K_M``
int8     : ``qwen3.5:4b-q8_0``
"""
from __future__ import annotations

from typing import Generator, List, Optional, Tuple

from src.models import EdgeRelationship, Movie

# Ollama is imported lazily so the app still loads when ollama is not installed
try:
    import ollama as _ollama_lib
    _OLLAMA_PKG_AVAILABLE = True
except ImportError:
    _OLLAMA_PKG_AVAILABLE = False

# ---------------------------------------------------------------------------
# Model name helpers
# ---------------------------------------------------------------------------

BASE_MODEL = "qwen3.5:4b"

_QUANT_SUFFIXES = {
    "default": "",
    "int4": "-q4_K_M",
    "int8": "-q8_0",
}


def resolve_model_name(base: str, quant: str = "default") -> str:
    """Return the full Ollama model tag for *base* and *quant*.

    Parameters
    ----------
    base:
        Base model name, e.g. ``"qwen3.5:4b"``.
    quant:
        One of ``"default"``, ``"int4"``, ``"int8"``.
    """
    suffix = _QUANT_SUFFIXES.get(quant, "")
    return f"{base}{suffix}"


# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------

def ollama_is_running(host: str = "http://localhost:11434") -> bool:
    """Return True if an Ollama server is reachable at *host*."""
    if not _OLLAMA_PKG_AVAILABLE:
        return False
    try:
        client = _ollama_lib.Client(host=host)
        client.list()
        return True
    except Exception:
        return False


def list_local_models(host: str = "http://localhost:11434") -> List[str]:
    """Return names of models pulled on the local Ollama server."""
    if not _OLLAMA_PKG_AVAILABLE:
        return []
    try:
        client = _ollama_lib.Client(host=host)
        return [m.model for m in client.list().models]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    path_data: List[Tuple[Movie, Movie, EdgeRelationship]],
) -> str:
    lines = [
        "You are a knowledgeable film critic. In 3–5 engaging sentences, explain "
        "why the following movies form a meaningful chain. Reference the specific "
        "signals that link each pair — shared audience taste, genre kinship, or "
        "thematic overlap. Write in second person. Be specific; avoid generic praise.\n"
    ]
    for i, (m_a, m_b, rel) in enumerate(path_data, 1):
        lines.append(
            f"Step {i}: {m_a.display_title()} → {m_b.display_title()}\n"
            f"  Connection: {rel.describe()}\n"
            f"  {m_a.display_title()} genres: {m_a.genre_str()}\n"
            f"  {m_b.display_title()} genres: {m_b.genre_str()}"
        )
        if rel.shared_tags:
            lines.append(f"  Shared viewer tags: {', '.join(rel.shared_tags[:5])}")
        lines.append("")
    return "\n".join(lines)


def _generate_kwargs(model: str, prompt: str, *, stream: bool) -> dict:
    """Build Ollama generate kwargs with model-specific toggles."""
    kwargs = {"model": model, "prompt": prompt, "stream": stream}
    # Qwen 3.5 can emit visible reasoning traces unless thinking is disabled.
    if model.strip().lower().startswith("qwen3.5"):
        kwargs["think"] = False
    return kwargs


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_story(
    path_data: List[Tuple[Movie, Movie, EdgeRelationship]],
    model: Optional[str] = None,
    host: str = "http://localhost:11434",
) -> str:
    """Generate a connection story.

    Parameters
    ----------
    path_data:
        Output of ``algorithms.path_relationships()``.
    model:
        Full Ollama model tag (e.g. ``"qwen3.5:4b"``).
        If *None*, the rule-based fallback is returned.
    host:
        Ollama server URL.
    """
    if model is None or not _OLLAMA_PKG_AVAILABLE:
        return _fallback_story(path_data)

    prompt = _build_prompt(path_data)
    try:
        client = _ollama_lib.Client(host=host)
        resp = client.generate(**_generate_kwargs(model, prompt, stream=False))
        return resp.response.strip()
    except Exception as exc:
        return _format_error(exc, model) + "\n\n" + _fallback_story(path_data)


def stream_story(
    path_data: List[Tuple[Movie, Movie, EdgeRelationship]],
    model: Optional[str] = None,
    host: str = "http://localhost:11434",
) -> Generator[str, None, None]:
    """Stream the connection story token by token for live display.

    Yields successive text chunks.  Falls back to a single-chunk rule-based
    story if the LLM is unavailable.
    """
    if model is None or not _OLLAMA_PKG_AVAILABLE:
        yield _fallback_story(path_data)
        return

    prompt = _build_prompt(path_data)
    try:
        client = _ollama_lib.Client(host=host)
        for chunk in client.generate(**_generate_kwargs(model, prompt, stream=True)):
            if chunk.response:
                yield chunk.response
    except Exception as exc:
        yield _format_error(exc, model) + "\n\n"
        yield _fallback_story(path_data)


def _format_error(exc: Exception, model: str) -> str:
    """Return a human-readable error with a fix hint."""
    msg = str(exc)
    if "404" in msg or "not found" in msg.lower():
        return (
            f"> **Model `{model}` is not downloaded yet.**\n"
            f"> Run this in your terminal, then try again:\n"
            f"> ```\n"
            f"> ollama pull {model}\n"
            f"> ```"
        )
    if "connection" in msg.lower() or "refused" in msg.lower():
        return (
            "> **Ollama server is not running.**\n"
            "> Start it with:\n"
            "> ```\n"
            "> ollama serve\n"
            "> ```"
        )
    return f"> **LLM error:** {exc}"


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _fallback_story(path_data: List[Tuple[Movie, Movie, EdgeRelationship]]) -> str:
    if not path_data:
        return "No path data to narrate."
    parts = [
        f"{m_a.display_title()} connects to {m_b.display_title()} "
        f"through {rel.describe()}."
        for m_a, m_b, rel in path_data
    ]
    header = (
        f"Your journey from **{path_data[0][0].display_title()}** to "
        f"**{path_data[-1][1].display_title()}** spans {len(path_data)} hop(s). "
    )
    return header + " ".join(parts)
