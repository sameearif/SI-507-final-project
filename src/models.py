"""Domain model classes for the Movie Discovery Graph."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Any


@dataclass
class Movie:
    """A movie node in the discovery graph.

    Attributes:
        movie_id: MovieLens integer ID.
        title: Cleaned title (year stripped).
        year: Release year parsed from MovieLens title string.
        genres: List of genre labels from MovieLens.
        imdb_id: IMDb tconst (e.g. 'tt0114709'), used to join IMDb data.
        avg_rating: Mean rating from MovieLens users (0 if unrated).
        rating_count: Number of MovieLens ratings.
        imdb_rating: IMDb weighted average rating (0 if unavailable).
        imdb_votes: IMDb vote count (0 if unavailable).
        tags: Set of lowercase user-generated tags from MovieLens.
    """

    movie_id: int
    title: str
    year: Optional[int]
    genres: List[str]
    imdb_id: Optional[str] = None
    avg_rating: float = 0.0
    rating_count: int = 0
    imdb_rating: float = 0.0
    imdb_votes: int = 0
    tags: Set[str] = field(default_factory=set)

    def __hash__(self) -> int:
        return hash(self.movie_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Movie) and self.movie_id == other.movie_id

    def __repr__(self) -> str:
        return f"Movie({self.movie_id}, '{self.display_title()}')"

    def display_title(self) -> str:
        """Return title with year appended in parentheses if available."""
        if self.year:
            return f"{self.title} ({self.year})"
        return self.title

    def genre_str(self) -> str:
        """Return genres as a comma-separated string."""
        return ", ".join(self.genres) if self.genres else "Unknown"

    def top_tags(self, n: int = 5) -> List[str]:
        """Return up to n tags sorted alphabetically (strings only)."""
        return sorted(t for t in self.tags if isinstance(t, str))[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary (JSON-safe)."""
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "year": self.year,
            "genres": self.genres,
            "imdb_id": self.imdb_id,
            "avg_rating": round(self.avg_rating, 2),
            "rating_count": self.rating_count,
            "imdb_rating": self.imdb_rating,
            "imdb_votes": self.imdb_votes,
            "tags": sorted(t for t in self.tags if isinstance(t, str)),
        }


@dataclass
class EdgeRelationship:
    """Captures why two movies are connected in the graph.

    The three signals are:
    - tag_similarity: Genome cosine similarity (ml-32m) or user-tag Jaccard.
    - audience_overlap: Jaccard overlap of users who rated each movie >= 3.5.
    - genre_similarity: Jaccard overlap of genre lists.

    weight is a weighted composite of the three signals and drives
    edge weight in the NetworkX graph.
    """

    tag_similarity: float = 0.0      # genome cosine when ml-32m is used
    audience_overlap: float = 0.0
    genre_similarity: float = 0.0
    shared_tags: List[str] = field(default_factory=list)
    weight: float = 0.0

    def describe(self) -> str:
        """Return a human-readable description of the relationship."""
        parts: List[str] = []
        if self.tag_similarity >= 0.5:
            parts.append(f"content similarity {self.tag_similarity:.0%}")
        if self.shared_tags:
            sample = ", ".join(f'"{t}"' for t in self.shared_tags[:3])
            more = f" +{len(self.shared_tags) - 3} more" if len(self.shared_tags) > 3 else ""
            parts.append(f"shared tags: {sample}{more}")
        if self.audience_overlap >= 0.05:
            parts.append(f"audience overlap {self.audience_overlap:.0%}")
        if self.genre_similarity >= 0.5:
            parts.append(f"genre match {self.genre_similarity:.0%}")
        return " • ".join(parts) if parts else "general similarity"

    def dominant_signal(self) -> str:
        """Return which of the three signals contributes most."""
        signals = {
            "shared tags": self.tag_similarity * 0.5,
            "audience overlap": self.audience_overlap * 0.3,
            "genre similarity": self.genre_similarity * 0.2,
        }
        return max(signals, key=signals.get)
