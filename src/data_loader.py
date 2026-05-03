"""Download and process MovieLens ml-32m and IMDb datasets.

Data sources
------------
Primary  : MovieLens ml-32m — ratings (32M), tags (2M+), genome scores, movies, links
Secondary: IMDb title.ratings.tsv.gz — external vote/rating data per IMDb title

The ratings.csv in ml-32m is ~1.2 GB; it is streamed in 1M-row chunks to
avoid loading it fully into memory.  Genome scores (16M rows, 14K movies ×
1128 tags) are loaded into a dense float32 matrix (~130 MB) and used as the
primary edge signal in the graph.
"""
from __future__ import annotations

import gzip
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.models import Movie

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"

# Folder name inside the zip
ML_DIRNAME = "ml-32m"

GRAPH_CACHE = PROCESSED_DIR / "graph_ml32m.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path, label: str = "") -> Path:
    if dest.exists():
        return dest
    print(f"Downloading {label or url} …")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=label or dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 17):  # 128 KB chunks
            fh.write(chunk)
            bar.update(len(chunk))
    return dest


def _parse_year(raw_title: str) -> Tuple[str, Optional[int]]:
    title = raw_title.strip()
    year: Optional[int] = None
    if title.endswith(")") and "(" in title:
        paren_start = title.rfind("(")
        candidate = title[paren_start + 1 : -1]
        if candidate.isdigit() and len(candidate) == 4:
            year = int(candidate)
            title = title[:paren_start].strip()
    return title, year


# ---------------------------------------------------------------------------
# Raw loaders
# ---------------------------------------------------------------------------

def _ml_dir() -> Path:
    _ensure_dirs()
    ml_dir = RAW_DIR / ML_DIRNAME
    if not ml_dir.exists():
        zip_path = RAW_DIR / "ml-32m.zip"
        _download(MOVIELENS_URL, zip_path, "MovieLens ml-32m (~290 MB)")
        print("Extracting archive …")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(RAW_DIR)
    return ml_dir


def load_movies_csv() -> pd.DataFrame:
    return pd.read_csv(_ml_dir() / "movies.csv")


def load_links_csv() -> pd.DataFrame:
    return pd.read_csv(_ml_dir() / "links.csv")


def load_tags_csv() -> pd.DataFrame:
    """Load user-generated tags (2M+ rows).  Used only for display metadata."""
    tags_file = _ml_dir() / "tags.csv"
    return pd.read_csv(tags_file, usecols=["userId", "movieId", "tag"])


def compute_rating_stats(
    min_ratings: int = 100,
) -> Dict[int, Tuple[float, int]]:
    """Stream ratings.csv in 1M-row chunks; return {movieId: (mean, count)}.

    Only movies that reach *min_ratings* are included, so this also acts as
    the eligibility filter for the graph.
    """
    ratings_file = _ml_dir() / "ratings.csv"
    print(f"Streaming ratings (this takes ~30–60 s for 32 M rows) …")
    total_rows = 0
    sum_r: Dict[int, float] = defaultdict(float)
    cnt_r: Dict[int, int] = defaultdict(int)

    for chunk in pd.read_csv(
        ratings_file,
        chunksize=1_000_000,
        usecols=["movieId", "rating"],
        dtype={"movieId": "int32", "rating": "float32"},
    ):
        total_rows += len(chunk)
        grouped = chunk.groupby("movieId", sort=False)["rating"]
        for mid, s in grouped.sum().items():
            sum_r[mid] += float(s)
        for mid, c in grouped.count().items():
            cnt_r[mid] += int(c)

    print(f"  Processed {total_rows:,} ratings for {len(cnt_r):,} movies")
    return {
        mid: (sum_r[mid] / cnt_r[mid], cnt_r[mid])
        for mid in cnt_r
        if cnt_r[mid] >= min_ratings
    }


def load_high_ratings(
    eligible_ids: set,
    min_rating: float = 3.5,
) -> Dict[int, set]:
    """Stream ratings.csv and return {movieId: set(userId)} for high ratings.

    Only considers *eligible_ids* to keep memory bounded.
    """
    ratings_file = _ml_dir() / "ratings.csv"
    print("Streaming ratings for audience matrix …")
    fans: Dict[int, set] = defaultdict(set)
    for chunk in pd.read_csv(
        ratings_file,
        chunksize=1_000_000,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    ):
        high = chunk[
            (chunk["rating"] >= min_rating) & (chunk["movieId"].isin(eligible_ids))
        ]
        for row in high.itertuples(index=False):
            fans[int(row.movieId)].add(int(row.userId))
    return dict(fans)


def load_genome() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load genome scores into a (movie_ids_array, matrix) pair.

    Returns
    -------
    movie_ids : np.ndarray shape (n_movies,) of int
    matrix    : np.ndarray shape (n_movies, 1128) of float32

    Returns None if genome files are absent (e.g. in the small dataset).
    """
    ml_dir = _ml_dir()
    scores_file = ml_dir / "genome-scores.csv"
    if not scores_file.exists():
        print("No genome-scores.csv found; skipping genome similarity.")
        return None

    print("Loading genome scores …")
    genome_df = pd.read_csv(
        scores_file,
        dtype={"movieId": "int32", "tagId": "int16", "relevance": "float32"},
    )

    # Pivot: rows = movies, cols = tags (1128 tags)
    pivot = genome_df.pivot(index="movieId", columns="tagId", values="relevance")
    movie_ids = pivot.index.to_numpy(dtype=np.int32)
    matrix = pivot.to_numpy(dtype=np.float32)  # shape (n_movies, 1128)
    print(f"  Genome matrix: {matrix.shape[0]:,} movies × {matrix.shape[1]} tags")
    return movie_ids, matrix


def load_imdb_ratings() -> pd.DataFrame:
    _ensure_dirs()
    gz_path = RAW_DIR / "title.ratings.tsv.gz"
    _download(IMDB_RATINGS_URL, gz_path, "IMDb Ratings (~8 MB)")
    with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(fh, sep="\t", dtype={"tconst": str})
    df["numVotes"] = pd.to_numeric(df["numVotes"], errors="coerce").fillna(0).astype(int)
    df["averageRating"] = pd.to_numeric(df["averageRating"], errors="coerce").fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Build Movie objects
# ---------------------------------------------------------------------------

def build_movies(
    rating_stats: Dict[int, Tuple[float, int]],
    min_ratings: int = 100,
) -> Dict[int, Movie]:
    """Construct the {movie_id: Movie} dict from all raw sources."""
    eligible_ids = set(rating_stats.keys())

    movies_df = load_movies_csv()
    links_df = load_links_csv()
    imdb_df = load_imdb_ratings()

    # IMDb join
    links_clean = links_df.dropna(subset=["imdbId"]).copy()
    links_clean["imdbId"] = (
        links_clean["imdbId"].astype(int).astype(str).str.zfill(7)
    )
    links_clean["tconst"] = "tt" + links_clean["imdbId"]
    mid_to_tconst = dict(
        zip(links_clean["movieId"].astype(int), links_clean["tconst"])
    )
    imdb_map = {
        row.tconst: (row.averageRating, row.numVotes)
        for row in imdb_df.itertuples()
    }

    # Tags for display (subset only)
    print("Loading tags for display metadata …")
    tags_df = load_tags_csv()
    # Drop rows where tag is NaN before any conversion so floats never enter sets
    tags_df = tags_df.dropna(subset=["tag"])
    tags_df["tag"] = tags_df["tag"].astype(str).str.lower().str.strip()
    # Exclude empty strings and the literal "nan" that astype produces for NA
    tags_df = tags_df[tags_df["tag"].str.len() > 0]
    tags_df = tags_df[tags_df["tag"] != "nan"]
    tags_by_movie: Dict[int, set] = (
        tags_df[tags_df["movieId"].isin(eligible_ids)]
        .groupby("movieId")["tag"]
        .apply(lambda ts: {t for t in ts if isinstance(t, str) and t})
        .to_dict()
    )

    movies: Dict[int, Movie] = {}
    for row in movies_df.itertuples():
        mid = int(row.movieId)
        if mid not in eligible_ids:
            continue

        avg_r, count = rating_stats[mid]
        title, year = _parse_year(row.title)
        genres = (
            [g for g in str(row.genres).split("|") if g and g != "(no genres listed)"]
            if row.genres
            else []
        )
        tconst = mid_to_tconst.get(mid)
        imdb_r, imdb_v = imdb_map.get(tconst, (0.0, 0)) if tconst else (0.0, 0)

        movies[mid] = Movie(
            movie_id=mid,
            title=title,
            year=year,
            genres=genres,
            imdb_id=tconst,
            avg_rating=float(avg_r),
            rating_count=int(count),
            imdb_rating=float(imdb_r),
            imdb_votes=int(imdb_v),
            tags=tags_by_movie.get(mid, set()),
        )

    return movies


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def load_all(min_ratings: int = 100) -> Dict[int, Movie]:
    """Download everything and return the {movie_id: Movie} dict.

    *min_ratings* controls how many MovieLens ratings a movie needs to be
    included.  Lower = more movies but sparser data; higher = fewer movies
    with richer connections.  100 is a good default for ml-32m.
    """
    rating_stats = compute_rating_stats(min_ratings=min_ratings)
    return build_movies(rating_stats, min_ratings=min_ratings)
