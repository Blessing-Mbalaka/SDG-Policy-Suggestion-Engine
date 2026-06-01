from __future__ import annotations

import html
from dataclasses import dataclass

from policy_recommendation_engine.embeddings import cosine_similarity
from policy_recommendation_engine.models import PipelineResult


@dataclass(frozen=True)
class GraphNode:
    index: int
    label: str
    theme: str
    preview: str


@dataclass(frozen=True)
class GraphEdge:
    left: int
    right: int
    similarity: float


@dataclass(frozen=True)
class EmbeddingGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    svg: str


def build_embedding_graph(result: PipelineResult, similarity_threshold: float = 0.35) -> EmbeddingGraph:
    nodes = build_nodes(result)
    edges = build_edges(result, similarity_threshold)
    svg = render_graph_svg(nodes, edges)
    return EmbeddingGraph(nodes=nodes, edges=edges, svg=svg)


def build_nodes(result: PipelineResult) -> tuple[GraphNode, ...]:
    theme_by_document = map_document_indexes_to_themes(result)
    nodes: list[GraphNode] = []

    for index, processed in enumerate(result.documents):
        preview = processed.document.text.replace("\n", " ").strip()
        if len(preview) > 90:
            preview = preview[:87] + "..."
        nodes.append(
            GraphNode(
                index=index,
                label=label_for_document(index, processed.document.metadata),
                theme=theme_by_document.get(index, "unclustered"),
                preview=preview,
            )
        )

    return tuple(nodes)


def map_document_indexes_to_themes(result: PipelineResult) -> dict[int, str]:
    theme_by_document: dict[int, str] = {}
    for theme in result.themes:
        for document_index in theme.document_indexes:
            theme_by_document[document_index] = theme.name
    return theme_by_document


def label_for_document(index: int, metadata: dict[str, object]) -> str:
    page_number = metadata.get("page_number")
    block_number = metadata.get("block_number")
    row_number = metadata.get("upload_row")

    if page_number:
        return f"Doc {index + 1} P{page_number}"
    if block_number:
        return f"Doc {index + 1} B{block_number}"
    if row_number:
        return f"Doc {index + 1} R{row_number}"
    return f"Doc {index + 1}"


def build_edges(result: PipelineResult, similarity_threshold: float) -> tuple[GraphEdge, ...]:
    edges: list[GraphEdge] = []
    vectors = result.embedding_vectors

    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            similarity = cosine_similarity(vectors[left], vectors[right])
            if len(vectors) <= 30 or similarity >= similarity_threshold:
                edges.append(GraphEdge(left=left, right=right, similarity=round(similarity, 3)))

    if not edges and len(vectors) > 1:
        edges = build_fallback_edges(vectors)

    return tuple(edges)


def build_fallback_edges(vectors: tuple[tuple[float, ...], ...]) -> list[GraphEdge]:
    scored_edges: list[GraphEdge] = []
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            similarity = cosine_similarity(vectors[left], vectors[right])
            scored_edges.append(GraphEdge(left=left, right=right, similarity=round(similarity, 3)))
    scored_edges.sort(key=lambda edge: edge.similarity, reverse=True)
    return scored_edges[: max(1, min(3, len(scored_edges)))]


def render_graph_svg(nodes: tuple[GraphNode, ...], edges: tuple[GraphEdge, ...]) -> str:
    if not nodes:
        return '<p class="hint">No embedding graph available.</p>'

    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("Embedding graph visualization requires NetworkX. Run: .\\.venv\\Scripts\\python -m pip install networkx") from exc

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.index)
    for edge in edges:
        graph.add_edge(edge.left, edge.right, weight=max(edge.similarity, 0.05))

    positions = nx.spring_layout(graph, seed=42, weight="weight") if len(nodes) > 1 else {0: (0.0, 0.0)}
    coordinates = scale_positions(positions)

    edge_markup = "".join(render_edge(edge, coordinates) for edge in edges)
    node_markup = "".join(render_node(node, coordinates[node.index]) for node in nodes)
    legend = render_legend(nodes)

    return (
        '<div class="graph-wrap">'
        '<svg class="embedding-graph" viewBox="0 0 900 420" role="img" aria-label="Embedding similarity network">'
        f"{edge_markup}{node_markup}"
        "</svg>"
        f"{legend}"
        "</div>"
    )


def scale_positions(positions: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
    x_values = [position[0] for position in positions.values()]
    y_values = [position[1] for position in positions.values()]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    width = max(max_x - min_x, 0.001)
    height = max(max_y - min_y, 0.001)

    scaled: dict[int, tuple[float, float]] = {}
    for node_index, position in positions.items():
        x = 70 + ((position[0] - min_x) / width) * 760
        y = 55 + ((position[1] - min_y) / height) * 300
        scaled[node_index] = (x, y)
    return scaled


def render_edge(edge: GraphEdge, coordinates: dict[int, tuple[float, float]]) -> str:
    left_x, left_y = coordinates[edge.left]
    right_x, right_y = coordinates[edge.right]
    width = 1.0 + max(edge.similarity, 0.0) * 3.0
    opacity = 0.25 + max(edge.similarity, 0.0) * 0.5
    return (
        f'<line x1="{left_x:.1f}" y1="{left_y:.1f}" x2="{right_x:.1f}" y2="{right_y:.1f}" '
        f'stroke="#78909c" stroke-width="{width:.2f}" opacity="{opacity:.2f}">'
        f"<title>Similarity {edge.similarity:.2f}</title>"
        "</line>"
    )


def render_node(node: GraphNode, coordinate: tuple[float, float]) -> str:
    x, y = coordinate
    color = color_for_theme(node.theme)
    safe_label = html.escape(node.label)
    safe_theme = html.escape(node.theme)
    safe_preview = html.escape(node.preview)
    return (
        f'<g class="graph-node">'
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="22" fill="{color}" stroke="#17323a" stroke-width="1.5">'
        f"<title>{safe_label}: {safe_theme} - {safe_preview}</title>"
        "</circle>"
        f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle">{safe_label}</text>'
        "</g>"
    )


def render_legend(nodes: tuple[GraphNode, ...]) -> str:
    themes = sorted({node.theme for node in nodes})
    items = "".join(
        f'<span><i style="background:{color_for_theme(theme)}"></i>{html.escape(theme)}</span>' for theme in themes
    )
    return f'<div class="graph-legend">{items}</div>'


def color_for_theme(theme: str) -> str:
    palette = ("#4f9d8f", "#d98055", "#7a8fdd", "#c08497", "#8aa05a", "#c9a227", "#6aa4c8", "#9c7fb3")
    index = sum(ord(character) for character in theme) % len(palette)
    return palette[index]
