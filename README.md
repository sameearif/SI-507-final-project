# Movie Connection & Discovery Graph

A Streamlit app that turns 32 million movie ratings into an interactive graph of cinema. Movies are nodes; edges encode shared audience taste, genome-based content similarity, and genre kinship. Explore neighbourhoods, find the shortest path between any two films, rank the most influential titles, and get graph-proximity recommendations — all from a clean web UI.

An optional local LLM (via Ollama) can narrate *why* two movies are connected in plain English. The app is fully functional without any API key or internet connection after the initial data download.

---

## Features

| Page | What you can do |
|---|---|
| **Search & Neighbourhood** | Find a movie, see its graph neighbourhood (depth 1 or 2), inspect direct connections |
| **Path Finder** | Shortest path between any two movies + optional AI-generated connection story |
| **Rankings** | Top movies by Degree / Betweenness / PageRank / Closeness centrality, with genre and year filters |
| **Recommendations** | Graph-proximity recommendations (not just ratings — structural closeness) |
| **Explore & Filter** | Filter by genre, year range, rating floor, and user-tag keyword; mini connection map |
| **Graph Statistics** | Degree distribution, genre breakdown, density, connected components |

---

## Data Sources

| Source | What it provides | Size |
|---|---|---|
| [MovieLens ml-32m](https://grouplens.org/datasets/movielens/32m/) (GroupLens) | 32 M ratings, 87 K movies, genome scores (1 128 tags × 14 K movies), user tags | ~290 MB zip |
| [IMDb title.ratings](https://developer.imdb.com/non-commercial-datasets/) | External average rating and vote count per title | ~8 MB gz |

Both files are downloaded automatically on first run and cached locally. You do not need to create any accounts.

### How graph edges are built

Each pair of movies gets a composite similarity score from three signals:

| Signal | Weight | Description |
|---|---|---|
| **Genome cosine similarity** | 0.50 | Cosine distance between 1 128-dimensional viewer-attribute vectors from the MovieLens genome |
| **Audience overlap** | 0.30 | Jaccard similarity of the sets of users who rated each movie ≥ 3.5 |
| **Genre similarity** | 0.20 | Jaccard similarity of genre label sets |

An edge is created when the composite score ≥ 0.12. Movies with fewer than 100 ratings are excluded so every node has meaningful data.

---

## Project Structure

```
Final Project/
├── app.py                  # Streamlit app (6 pages)
├── requirements.txt
├── .env.example
├── src/
│   ├── models.py           # Movie, EdgeRelationship dataclasses
│   ├── data_loader.py      # Download + process MovieLens ml-32m & IMDb
│   ├── graph_builder.py    # Build NetworkX graph (vectorised scipy + sklearn)
│   ├── algorithms.py       # Search, filter, path finding, centrality, recommendations
│   ├── llm.py              # Optional Ollama connection-story generation
│   └── visualization.py   # Plotly graph and chart helpers
├── tests/
│   ├── test_models.py      # 18 tests — Movie and EdgeRelationship
│   ├── test_algorithms.py  # 49 tests — search, path, centrality, recommendations
│   └── test_graph.py       # 33 tests — similarity matrices and graph assembly
└── data/
    ├── raw/                # Downloaded datasets (auto-created)
    └── processed/          # Cached graph pickle (auto-created)
```

---

## Setup

### 1. Prerequisites

- Python 3.11 or later (tested on 3.14.4)
- macOS, Linux, or Windows
- ~2 GB free disk space for datasets and cache

### 2. Create and activate a virtual environment

```bash
# Create the environment
python3 -m venv env

# Activate — macOS / Linux
source env/bin/activate

# Activate — Windows (PowerShell)
.\env\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
# Make sure the venv is active
source env/bin/activate   # macOS / Linux

streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

### First run

On first launch the app will:

1. Download **MovieLens ml-32m** (~290 MB) and extract it
2. Download **IMDb title.ratings.tsv.gz** (~8 MB)
3. Stream all 32 million ratings to compute per-movie statistics (~30–60 s)
4. Load genome scores and build the similarity graph (~2–5 min)
5. Cache the graph to `data/processed/graph_ml32m.pkl`

Subsequent starts skip all of that and load the cache in a few seconds.

> The first-run progress is printed to the terminal, not the browser. Keep the terminal visible.

---

## Optional: Connection Stories with Ollama (Local LLM)

The Path Finder page can narrate *why* two movies are connected using a local language model. This is entirely optional — every other feature works without it.

> **Important:** Ollama has two separate components that are often confused:
> - The **Ollama app / server** — a native binary installed on your OS. This is what runs models. `pip install ollama` does **not** install this.
> - The **`ollama` Python package** — the client library this project uses to talk to the server. It is already in `requirements.txt` and installed when you ran `pip install -r requirements.txt`.
>
> You need both. Install the server first (steps below), then the Python package is already done.

---

### Step 1 — Install the Ollama server (native app)

The Ollama server is a standalone binary. Install it with one of the following methods:

**macOS — Homebrew (recommended)**
```bash
brew install ollama
```

**macOS — Official installer**

Download the `.dmg` from [ollama.com/download](https://ollama.com/download), open it, and drag Ollama to Applications.

**Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**

Download the installer from [ollama.com/download](https://ollama.com/download) and run it.

After installation, verify it works:
```bash
ollama --version
# Should print something like: ollama version 0.x.x
```

If you see `zsh: command not found: ollama`, the server is not installed — `pip install ollama` only installed the Python client, not the server. Repeat the step above.

---

### Step 2 — Start the Ollama server

```bash
ollama serve
```

Leave this running in a separate terminal. On macOS, if you installed via the `.dmg`, the Ollama app may start the server automatically when launched from Applications.

---

### Step 3 — Pull the model

Choose one quantisation based on your available RAM:

| Quantisation | Command | RAM needed | Notes |
|---|---|---|---|
| Explicit int4 | `ollama pull qwen3.5:4b-q4_K_M` | ~3 GB | Faster, lower quality |
| int8 | `ollama pull qwen3.5:4b-q8_0` | ~5 GB | Slower, better quality |

The download is ~2–3 GB. You only need to do this once.

```bash
# Confirm the model is available after pulling
ollama list
```

---

### Step 4 — Enable in the app

With `ollama serve` running and the model pulled, open the Streamlit app and in the **sidebar**:

1. Set **Connection Story (LLM)** → `Ollama (local)`
2. Check that the **Model name** field matches what you pulled (default: `qwen3.5:4b`)
3. Select your **Quantisation** from the dropdown (`int4` is the default in the app)
4. A green **"Ollama reachable"** badge confirms the server is connected

You can use any other model you have pulled — just type its full Ollama tag in the Model name field (e.g. `llama3.2:3b`, `mistral:7b`).

---

### Quantisation quick reference

```
qwen3.5:4b              ← default (Ollama picks, usually q4_K_M)
qwen3.5:4b-q4_K_M      ← explicit int4  (~3 GB RAM)
qwen3.5:4b-q8_0        ← int8           (~5 GB RAM)
```

If Ollama is not running or the model is not pulled, the app falls back to a rule-based connection summary automatically — no crash, no error dialog.

---

### Troubleshooting

| Symptom | Fix |
|---|---|
| `zsh: command not found: ollama` | The server binary is not installed — run `brew install ollama` or download from ollama.com |
| Sidebar shows "Ollama not reachable" | Run `ollama serve` in a separate terminal and keep it open |
| `pull` fails | Check your internet connection; the model download is ~2–3 GB |
| Model generates slowly | Switch to int4 quantisation; close other RAM-heavy apps |

---

## Running the Tests

```bash
# From the project root with the venv active
python -m pytest tests/ -v
```

Expected output: **100 tests, all passing** in under 15 seconds.

The test suite covers:
- `test_models.py` — Movie construction, equality, hashing, tag methods, EdgeRelationship signals
- `test_algorithms.py` — Search, filter, neighbourhood BFS, path finding, centrality, recommendations, graph stats
- `test_graph.py` — Genome cosine similarity, audience Jaccard, genre Jaccard, full graph assembly (all using synthetic in-memory data, no downloads required)

---

## Technical Notes

### Memory usage

| Stage | Approximate RAM |
|---|---|
| Rating stats streaming (32 M rows) | < 200 MB peak |
| Genome matrix (14 K × 1 128 floats) | ~130 MB |
| Audience Jaccard matrix | ~400 MB peak |
| Loaded graph (NetworkX pickle) | ~200–400 MB |
| Ollama model (int4) | ~3 GB |

A machine with 8 GB RAM is sufficient for everything including the LLM.

### Graph size (typical, min_ratings = 100)

| Metric | Value |
|---|---|
| Movies (nodes) | ~10 000–15 000 |
| Connections (edges) | ~200 000–500 000 |
| Connected components | 1–3 |
| Average degree | ~30–80 |

### Changing the minimum rating threshold

`min_ratings` controls how many MovieLens ratings a movie needs to appear in the graph. Lower = more movies, sparser data. Higher = fewer movies, richer connections. To change it, edit line 70 of `app.py`:

```python
@st.cache_resource(show_spinner="Loading movie data from ml-32m…")
def _load_movies() -> Dict[int, Movie]:
    return load_all(min_ratings=100)   # change this value
```

Then delete `data/processed/graph_ml32m.pkl` to force a rebuild.

---

## Acknowledgements

- [MovieLens](https://grouplens.org/datasets/movielens/) datasets by GroupLens Research, University of Minnesota
- [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/) by IMDb
- [Ollama](https://ollama.com) for local model serving
- [Qwen 3.5](https://ollama.com/library/qwen3.5) by Alibaba Cloud
