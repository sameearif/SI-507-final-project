"""Plotly-based graph visualisation helpers for the Streamlit app."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import plotly.graph_objects as go

from src.models import Movie

# Colour palette for genres
_GENRE_COLOURS = {
    "Action": "#e74c3c",
    "Adventure": "#e67e22",
    "Animation": "#f1c40f",
    "Comedy": "#2ecc71",
    "Crime": "#8e44ad",
    "Documentary": "#95a5a6",
    "Drama": "#3498db",
    "Fantasy": "#1abc9c",
    "Horror": "#c0392b",
    "Musical": "#d35400",
    "Mystery": "#6c3483",
    "Romance": "#e91e8c",
    "Sci-Fi": "#0d6efd",
    "Thriller": "#2c3e50",
    "War": "#7f8c8d",
    "Western": "#a04000",
}
_DEFAULT_COLOUR = "#aaaaaa"


def _movie_colour(movie: Movie) -> str:
    for genre in movie.genres:
        if genre in _GENRE_COLOURS:
            return _GENRE_COLOURS[genre]
    return _DEFAULT_COLOUR


def draw_neighborhood(
    subgraph: nx.Graph,
    center_id: int,
    height: int = 500,
) -> go.Figure:
    """Draw a neighbourhood graph centred on *center_id*.

    Uses a spring layout and colour-codes nodes by primary genre.
    """
    if subgraph.number_of_nodes() == 0:
        return go.Figure().update_layout(
            title="No connections found", height=height
        )

    pos = nx.spring_layout(subgraph, seed=42, weight="weight")

    edge_x: List[float] = []
    edge_y: List[float] = []
    for u, v in subgraph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#cccccc"),
        hoverinfo="none",
    )

    node_x, node_y, node_text, node_hover, node_colours, node_sizes = (
        [], [], [], [], [], []
    )
    for node_id in subgraph.nodes():
        movie: Movie = subgraph.nodes[node_id]["movie"]
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        node_text.append(movie.title[:20])
        node_hover.append(
            f"<b>{movie.display_title()}</b><br>"
            f"Genres: {movie.genre_str()}<br>"
            f"Rating: {movie.avg_rating:.1f} ({movie.rating_count:,} votes)"
        )
        node_colours.append(_movie_colour(movie))
        node_sizes.append(20 if node_id == center_id else 12)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        marker=dict(
            color=node_colours,
            size=node_sizes,
            line=dict(width=1, color="#ffffff"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(l=0, r=0, t=30, b=0),
            height=height,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#ffffff",
        ),
    )
    return fig


def draw_path(
    G: nx.Graph,
    path: List[int],
    height: int = 400,
) -> go.Figure:
    """Draw a linear path graph highlighting each hop."""
    if len(path) < 2:
        return go.Figure()

    # Linear layout: evenly spaced
    pos = {mid: (i, 0) for i, mid in enumerate(path)}

    edge_x, edge_y = [], []
    for u, v in zip(path[:-1], path[1:]):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=3, color="#3498db"), hoverinfo="none",
    )

    node_x, node_y, labels, hover = [], [], [], []
    for mid in path:
        movie: Movie = G.nodes[mid]["movie"]
        x, y = pos[mid]
        node_x.append(x)
        node_y.append(y)
        labels.append(f"<b>{movie.title[:18]}</b><br>({movie.year or '?'})")
        hover.append(
            f"<b>{movie.display_title()}</b><br>"
            f"Genres: {movie.genre_str()}<br>"
            f"Rating: {movie.avg_rating:.1f}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        hoverinfo="text", hovertext=hover,
        text=labels, textposition="top center",
        marker=dict(
            size=18, color="#e74c3c",
            line=dict(width=2, color="#ffffff"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False, hovermode="closest",
            margin=dict(l=20, r=20, t=20, b=60),
            height=height,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-0.5, 0.8]),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#ffffff",
        ),
    )
    return fig


def draw_rankings(
    ranked: List[Tuple[Movie, float]],
    metric_label: str,
    height: int = 500,
) -> go.Figure:
    """Horizontal bar chart for centrality rankings."""
    movies = [m for m, _ in ranked]
    scores = [s for _, s in ranked]
    labels = [m.display_title()[:35] for m in movies]
    colours = [_movie_colour(m) for m in movies]

    fig = go.Figure(
        go.Bar(
            x=scores[::-1],
            y=labels[::-1],
            orientation="h",
            marker_color=colours[::-1],
            hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top Movies by {metric_label}",
        xaxis_title=metric_label,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#ffffff",
    )
    return fig
