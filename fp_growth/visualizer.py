"""
Visualization tools for FP-Tree and Association Rules using NetworkX and Matplotlib.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for CLI and Web servers
import matplotlib.pyplot as plt
import networkx as nx
from .fptree import FPTree, FPNode
from .rules import AssociationRule
from .miner import FrequentItemset


def draw_fp_tree(tree: FPTree, max_depth: int = 6) -> plt.Figure:
    """
    Generate a NetworkX visualization of the FP-tree with node counts and tree hierarchy.
    """
    G = nx.DiGraph()
    labels: Dict[int, str] = {}
    node_colors: List[str] = []
    node_sizes: List[int] = []

    node_counter = 0

    def traverse(node: FPNode, parent_id: Optional[int] = None, depth: int = 0):
        nonlocal node_counter
        if depth > max_depth:
            return

        curr_id = node_counter
        node_counter += 1

        if node.is_root:
            label = "ROOT"
            color = "#4A5568"
            size = 1400
        else:
            label = f"{node.item}\n(x{node.count})"
            color = "#3182CE"
            size = min(2200, 700 + node.count * 120)

        G.add_node(curr_id, label=label, depth=depth)
        labels[curr_id] = label
        node_colors.append(color)
        node_sizes.append(size)

        if parent_id is not None:
            G.add_edge(parent_id, curr_id)

        for child in node.children.values():
            traverse(child, curr_id, depth + 1)

    traverse(tree.root)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.set_title("FP-Tree Structural Topology & Node Frequencies", fontsize=14, fontweight="bold", pad=12)

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty FP-Tree", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Custom hierarchical layout based on node depth
    try:
        # Group nodes by depth for stratified layout
        depths: Dict[int, List[int]] = {}
        for n, data in G.nodes(data=True):
            d = data.get("depth", 0)
            depths.setdefault(d, []).append(n)

        pos: Dict[int, Tuple[float, float]] = {}
        for d, node_list in depths.items():
            width = len(node_list)
            for idx, n in enumerate(node_list):
                x = (idx + 1) / (width + 1)
                y = 1.0 - (d * 0.22)
                pos[n] = (x, y)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#A0AEC0",
        arrows=True,
        arrowsize=14,
        arrowstyle="-|>",
        width=1.5
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=8,
        font_family="sans-serif",
        font_color="white",
        font_weight="bold"
    )

    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_association_rules_network(rules: List[AssociationRule], top_n: int = 15) -> plt.Figure:
    """
    Visualizes association rules as a directed bipartite graph:
    Antecedent item -> Rule node -> Consequent item.
    """
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    ax.set_title(f"Top {min(len(rules), top_n)} Association Rules Directed Graph", fontsize=14, fontweight="bold", pad=12)

    if not rules:
        ax.text(0.5, 0.5, "No Association Rules Found with Current Thresholds", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    G = nx.DiGraph()
    subset_rules = rules[:top_n]

    item_nodes = set()
    rule_nodes = []

    for idx, rule in enumerate(subset_rules):
        rule_name = f"R{idx+1}\n(Lift: {rule.lift:.2f})"
        rule_nodes.append(rule_name)
        G.add_node(rule_name, node_type="rule", lift=rule.lift)

        for item in rule.antecedent:
            item_nodes.add(item)
            G.add_node(item, node_type="item")
            G.add_edge(item, rule_name)

        for item in rule.consequent:
            item_nodes.add(item)
            G.add_node(item, node_type="item")
            G.add_edge(rule_name, item)

    pos = nx.spring_layout(G, k=0.7, seed=42)

    # Node coloring by type
    item_color = "#38A169"  # Green
    rule_color = "#DD6B20"  # Orange

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(item_nodes),
        node_color=item_color,
        node_size=1200,
        alpha=0.92,
        ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=rule_nodes,
        node_color=rule_color,
        node_size=1400,
        node_shape="s",  # Squares for rules
        alpha=0.92,
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        edge_color="#718096",
        arrows=True,
        arrowsize=14,
        arrowstyle="-|>",
        width=1.8,
        ax=ax
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_color="white",
        font_weight="bold",
        ax=ax
    )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Transaction Items',
                   markerfacecolor=item_color, markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', label='Rule (Antecedent -> Consequent)',
                   markerfacecolor=rule_color, markersize=12),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)

    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_rules_scatter(rules: List[AssociationRule]) -> plt.Figure:
    """
    Scatter plot comparing Rule Support vs Confidence, colored by Lift.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.set_title("Association Rules: Support vs. Confidence (Colored by Lift)", fontsize=13, fontweight="bold", pad=10)

    if not rules:
        ax.text(0.5, 0.5, "No Association Rules Available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    supports = [r.support for r in rules]
    confidences = [r.confidence for r in rules]
    lifts = [r.lift for r in rules]

    scatter = ax.scatter(
        supports,
        confidences,
        c=lifts,
        cmap="plasma",
        s=120,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.6
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Lift", fontsize=11, fontweight="bold")

    ax.set_xlabel("Support", fontsize=11, fontweight="bold")
    ax.set_ylabel("Confidence", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig
