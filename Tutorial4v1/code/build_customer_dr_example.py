from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

SEGMENT_COLORS = {
    "budget": "#1f77b4",
    "premium": "#d62728",
    "family": "#2ca02c",
}
COMMUNITY_COLORS = {
    "north_circle": "#9467bd",
    "south_circle": "#ff7f0e",
}
SEGMENT_MARKERS = {
    "budget": "o",
    "premium": "s",
    "family": "^",
}
COMMUNITY_MARKERS = {
    "north_circle": "o",
    "south_circle": "s",
}


def build_customers() -> pd.DataFrame:
    rows = [
        {"customer_id": "C01", "age": 24, "income_k": 39, "tenure_months": 8, "region": "coast", "channel_pref": "store", "feature_segment": "budget", "social_circle": "north_circle"},
        {"customer_id": "C02", "age": 27, "income_k": 43, "tenure_months": 10, "region": "city", "channel_pref": "store", "feature_segment": "budget", "social_circle": "north_circle"},
        {"customer_id": "C03", "age": 25, "income_k": 41, "tenure_months": 7, "region": "suburb", "channel_pref": "store", "feature_segment": "budget", "social_circle": "south_circle"},
        {"customer_id": "C04", "age": 29, "income_k": 46, "tenure_months": 11, "region": "coast", "channel_pref": "web", "feature_segment": "budget", "social_circle": "south_circle"},
        {"customer_id": "C05", "age": 34, "income_k": 108, "tenure_months": 22, "region": "city", "channel_pref": "app", "feature_segment": "premium", "social_circle": "north_circle"},
        {"customer_id": "C06", "age": 37, "income_k": 115, "tenure_months": 27, "region": "suburb", "channel_pref": "app", "feature_segment": "premium", "social_circle": "north_circle"},
        {"customer_id": "C07", "age": 35, "income_k": 111, "tenure_months": 24, "region": "coast", "channel_pref": "app", "feature_segment": "premium", "social_circle": "south_circle"},
        {"customer_id": "C08", "age": 39, "income_k": 121, "tenure_months": 29, "region": "city", "channel_pref": "web", "feature_segment": "premium", "social_circle": "south_circle"},
        {"customer_id": "C09", "age": 43, "income_k": 76, "tenure_months": 40, "region": "suburb", "channel_pref": "web", "feature_segment": "family", "social_circle": "north_circle"},
        {"customer_id": "C10", "age": 46, "income_k": 83, "tenure_months": 45, "region": "coast", "channel_pref": "web", "feature_segment": "family", "social_circle": "north_circle"},
        {"customer_id": "C11", "age": 44, "income_k": 79, "tenure_months": 41, "region": "city", "channel_pref": "store", "feature_segment": "family", "social_circle": "south_circle"},
        {"customer_id": "C12", "age": 48, "income_k": 86, "tenure_months": 47, "region": "suburb", "channel_pref": "web", "feature_segment": "family", "social_circle": "south_circle"},
    ]
    return pd.DataFrame(rows)


def build_transactions(customers: pd.DataFrame) -> pd.DataFrame:
    segment_patterns = {
        "budget": [
            (1, "groceries", 42),
            (1, "fashion", 28),
            (2, "groceries", 47),
            (2, "home", 36),
            (3, "groceries", 45),
            (3, "electronics", 30),
        ],
        "premium": [
            (1, "electronics", 190),
            (1, "fashion", 150),
            (2, "electronics", 205),
            (2, "home", 95),
            (3, "fashion", 165),
            (3, "electronics", 215),
        ],
        "family": [
            (1, "groceries", 110),
            (1, "home", 92),
            (2, "groceries", 115),
            (2, "electronics", 70),
            (3, "home", 88),
            (3, "groceries", 120),
        ],
    }
    rows = []
    for _, customer in customers.iterrows():
        cid_num = int(customer["customer_id"][1:])
        customer_shift = (cid_num % 3) * 4
        for month, category, base_amount in segment_patterns[customer["feature_segment"]]:
            amount = base_amount + customer_shift
            rows.append(
                {
                    "customer_id": customer["customer_id"],
                    "month": month,
                    "category": category,
                    "amount": amount,
                }
            )
    return pd.DataFrame(rows)


def build_interactions() -> pd.DataFrame:
    edges = [
        ("C01", "C02"), ("C01", "C05"), ("C01", "C09"), ("C02", "C06"), ("C02", "C10"),
        ("C05", "C06"), ("C05", "C09"), ("C06", "C09"), ("C06", "C10"), ("C09", "C10"),
        ("C03", "C04"), ("C03", "C07"), ("C03", "C11"), ("C04", "C08"), ("C04", "C12"),
        ("C07", "C08"), ("C07", "C11"), ("C08", "C11"), ("C08", "C12"), ("C11", "C12"),
        ("C02", "C03"), ("C10", "C11"),
    ]
    return pd.DataFrame(edges, columns=["customer_i", "customer_j"])


def build_feature_matrix(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    spend_by_category = (
        transactions.pivot_table(
            index="customer_id",
            columns="category",
            values="amount",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns=lambda c: f"spend_{c}")
        .reset_index()
    )
    txn_summary = (
        transactions.groupby("customer_id")
        .agg(total_spend=("amount", "sum"), avg_transaction=("amount", "mean"))
        .reset_index()
    )
    feature_df = customers.merge(txn_summary, on="customer_id").merge(spend_by_category, on="customer_id")
    ordered_cols = [
        "customer_id",
        "feature_segment",
        "social_circle",
        "age",
        "income_k",
        "tenure_months",
        "total_spend",
        "avg_transaction",
        "spend_electronics",
        "spend_fashion",
        "spend_groceries",
        "spend_home",
    ]
    return feature_df[ordered_cols].sort_values("customer_id").reset_index(drop=True)


def zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    std = frame.std(ddof=0).replace(0, 1)
    return (frame - frame.mean()) / std


def truncated_svd_embedding(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    numeric_cols = [
        "age",
        "income_k",
        "tenure_months",
        "total_spend",
        "avg_transaction",
        "spend_electronics",
        "spend_fashion",
        "spend_groceries",
        "spend_home",
    ]
    x = zscore_frame(feature_df[numeric_cols]).to_numpy(dtype=float)
    u, singular_values, _ = np.linalg.svd(x, full_matrices=False)
    embedding = u[:, :2] * singular_values[:2]
    explained_ratio = (singular_values[:2] ** 2) / np.sum(singular_values ** 2)
    return embedding, explained_ratio, singular_values


def laplacian_eigenmaps_embedding(customers: pd.DataFrame, interactions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    node_order = customers["customer_id"].tolist()
    graph = nx.Graph()
    graph.add_nodes_from(node_order)
    graph.add_edges_from(interactions.itertuples(index=False, name=None))
    adjacency = nx.to_numpy_array(graph, nodelist=node_order, dtype=float)
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    embedding = eigenvectors[:, 1:3]
    return embedding, adjacency, laplacian, eigenvalues, eigenvectors, node_order


def save_tables(customers: pd.DataFrame, transactions: pd.DataFrame, interactions: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    transactions.to_csv(DATA_DIR / "transactions.csv", index=False)
    interactions.to_csv(DATA_DIR / "interactions.csv", index=False)
    feature_df.to_csv(DATA_DIR / "customer_feature_matrix.csv", index=False)


def write_data_description() -> None:
    description = """Agent-facing note for the synthetic customer dimensionality-reduction example

Use this file first when you need to understand or modify the example.
The goal of the example is not realism for its own sake; the goal is to produce a small dataset whose structure is easy to explain on slides and whose two low-dimensional views are deliberately different.

Files
- customers.csv: one row per customer.
- transactions.csv: six purchase rows per customer; used to derive the feature matrix.
- interactions.csv: customer-customer interaction links; used to build the graph for Laplacian Eigenmaps.
- customer_feature_matrix.csv: aggregated numeric matrix X used for truncated SVD.

Why this data was designed this way
- The same 12 customers are used in both methods so the comparison is about the method, not about changing datasets.
- The feature matrix X was designed to emphasize three behavior groups:
  budget, premium, family.
- The interaction graph was designed to emphasize two network communities:
  north_circle, south_circle.
- This mismatch is intentional.
  Truncated SVD should mainly recover the behavior groups because it only sees X.
  Laplacian Eigenmaps should mainly recover the network communities because it only sees the graph.
- The example is deliberately small enough to fit into one or two setup slides:
  12 customers, 72 transactions, and 22 interaction links.

Important columns in customers.csv
- customer_id:
  Stable customer identifier.
- age:
  Customer age in years.
- income_k:
  Approximate annual income in thousands.
- tenure_months:
  Number of months since the customer joined the business or loyalty program.
- region:
  Coarse geographic area of the customer: coast, city, or suburb.
  It is part of the raw relational dataset, but it is not used in the final SVD matrix so that the example stays small.
- channel_pref:
  Preferred shopping channel: store, web, or app.
  It indicates how the customer usually shops; it is kept in the raw table for interpretation.
- feature_segment:
  Synthetic behavior label used to generate purchase patterns.
  budget customers spend less and buy mostly groceries, premium customers spend more and buy more electronics/fashion, family customers concentrate on groceries/home.
  This is used only to color or interpret plots and to validate the SVD example; it is not given to either dimensionality-reduction method.
- social_circle:
  Synthetic community label used to generate the interaction graph.
  north_circle and south_circle indicate which part of the customer network the customer belongs to.
  This is used only for interpretation, plot coloring, and validation; it is not part of X.

Important columns in transactions.csv
- month:
  Small time index used only to create repeated purchases for each customer.
- category:
  Purchase category: electronics, fashion, groceries, or home.
- amount:
  Transaction amount.

Important columns in customer_feature_matrix.csv
- total_spend:
  Sum of transaction amounts for the customer.
- avg_transaction:
  Mean transaction amount for the customer.
- spend_electronics / spend_fashion / spend_groceries / spend_home:
  Total spend of the customer in each purchase category.

How the example demonstrates the two methods
- Truncated SVD input:
  customer_feature_matrix.csv
  The main signal is behavioral similarity in the engineered numeric features.
- Laplacian Eigenmaps input:
  interactions.csv transformed into an adjacency matrix A and Laplacian L = D - A.
  The main signal is local neighborhood structure in the customer graph.
- Expected teaching outcome:
  SVD groups customers that buy similarly, even if they are in different social circles.
  Laplacian Eigenmaps groups customers that are connected in the graph, even if they have different purchase behavior.
- The difference between the two outputs is therefore the core lesson:
  each method preserves a different notion of proximity.

What a coding agent should preserve if modifying the example
- Keep the example small.
- Keep the feature-group signal strong enough that SVD separates the three feature segments.
- Keep the graph-community signal strong enough that Laplacian Eigenmaps separates the two social circles.
- Avoid adding extra columns or graph complexity unless they support a specific slide.
"""
    with open(DATA_DIR / "data_description.txt", "w", encoding="utf-8") as handle:
        handle.write(description)


def make_embedding_df(feature_df: pd.DataFrame, coords: np.ndarray, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": feature_df["customer_id"],
            "feature_segment": feature_df["feature_segment"],
            "social_circle": feature_df["social_circle"],
            f"{prefix}_x": coords[:, 0],
            f"{prefix}_y": coords[:, 1],
        }
    )


def separation_score(df: pd.DataFrame, x_col: str, y_col: str, label_col: str) -> float:
    points = df[[x_col, y_col]].to_numpy(dtype=float)
    labels = df[label_col].to_numpy()
    unique_labels = sorted(set(labels))
    centroids = {label: points[labels == label].mean(axis=0) for label in unique_labels}

    within_group = np.mean(
        [np.linalg.norm(point - centroids[label]) for point, label in zip(points, labels)]
    )
    between_group = []
    for i, first_label in enumerate(unique_labels):
        for second_label in unique_labels[i + 1 :]:
            between_group.append(
                np.linalg.norm(centroids[first_label] - centroids[second_label])
            )
    return float(np.mean(between_group) / max(within_group, 1e-12))


def scatter_by_label(ax, df: pd.DataFrame, x_col: str, y_col: str, color_field: str, marker_field: str, title: str, color_map: dict[str, str], marker_map: dict[str, str]) -> None:
    for _, row in df.iterrows():
        ax.scatter(
            row[x_col],
            row[y_col],
            s=120,
            c=color_map[row[color_field]],
            marker=marker_map[row[marker_field]],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        ax.text(row[x_col] + 0.03, row[y_col] + 0.03, row["customer_id"], fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.axvline(0, color="#cccccc", linewidth=0.8)
    ax.grid(alpha=0.2)


def plot_embeddings(svd_df: pd.DataFrame, lap_df: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    scatter_by_label(
        axes[0],
        svd_df,
        "svd_x",
        "svd_y",
        color_field="feature_segment",
        marker_field="social_circle",
        title="Truncated SVD on feature matrix X\nColor = feature segment, marker = social circle",
        color_map=SEGMENT_COLORS,
        marker_map=COMMUNITY_MARKERS,
    )
    axes[0].set_xlabel("SVD dimension 1")
    axes[0].set_ylabel("SVD dimension 2")

    scatter_by_label(
        axes[1],
        lap_df,
        "laplacian_x",
        "laplacian_y",
        color_field="social_circle",
        marker_field="feature_segment",
        title="Laplacian Eigenmaps on interaction graph\nColor = social circle, marker = feature segment",
        color_map=COMMUNITY_COLORS,
        marker_map=SEGMENT_MARKERS,
    )
    axes[1].set_xlabel("Eigenvector 2")
    axes[1].set_ylabel("Eigenvector 3")

    fig.suptitle("Same customers, two different low-dimensional views", fontsize=14)
    fig.savefig(OUTPUT_DIR / "embedding_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    scatter_by_label(
        ax,
        svd_df,
        "svd_x",
        "svd_y",
        color_field="feature_segment",
        marker_field="social_circle",
        title="Truncated SVD embedding",
        color_map=SEGMENT_COLORS,
        marker_map=COMMUNITY_MARKERS,
    )
    ax.set_xlabel("SVD dimension 1")
    ax.set_ylabel("SVD dimension 2")
    fig.savefig(OUTPUT_DIR / "svd_embedding.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    scatter_by_label(
        ax,
        lap_df,
        "laplacian_x",
        "laplacian_y",
        color_field="social_circle",
        marker_field="feature_segment",
        title="Laplacian Eigenmaps embedding",
        color_map=COMMUNITY_COLORS,
        marker_map=SEGMENT_MARKERS,
    )
    ax.set_xlabel("Eigenvector 2")
    ax.set_ylabel("Eigenvector 3")
    fig.savefig(OUTPUT_DIR / "laplacian_embedding.png", dpi=200)
    plt.close(fig)


def plot_graph(customers: pd.DataFrame, interactions: pd.DataFrame) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(customers["customer_id"])
    graph.add_edges_from(interactions.itertuples(index=False, name=None))
    pos = nx.spring_layout(graph, seed=4)

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    nx.draw_networkx_edges(graph, pos, width=1.4, alpha=0.5, ax=ax)

    for community, group in customers.groupby("social_circle"):
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=group["customer_id"].tolist(),
            node_color=COMMUNITY_COLORS[community],
            edgecolors="black",
            node_size=700,
            linewidths=0.8,
            ax=ax,
        )
    nx.draw_networkx_labels(graph, pos, font_size=9, ax=ax)
    ax.set_title("Interaction graph used for Laplacian Eigenmaps\nColor = social circle")
    ax.axis("off")
    fig.savefig(OUTPUT_DIR / "interaction_graph.png", dpi=200)
    plt.close(fig)

def plot_compact_relation_graph(customers: pd.DataFrame, interactions: pd.DataFrame) -> None:
    # Compact customer-only graph for a pre-embedding slide.
    # Uses a reduced edge set and neutral styling to avoid visually obvious circles.
    selected_pairs = [
        ("C01", "C02"),
        ("C01", "C05"),
        ("C02", "C06"),
        ("C05", "C09"),
        ("C09", "C10"),
        ("C03", "C04"),
        ("C03", "C07"),
        ("C04", "C12"),
        ("C07", "C08"),
        ("C08", "C11"),
        ("C02", "C03"),
        ("C10", "C11"),
    ]
    available_pairs = set(interactions.itertuples(index=False, name=None))
    selected_pairs = [pair for pair in selected_pairs if pair in available_pairs]

    graph = nx.Graph()
    customer_ids = customers["customer_id"].tolist()
    graph.add_nodes_from(customer_ids)
    graph.add_edges_from(selected_pairs)

    rng = np.random.default_rng(17)
    base_pos = nx.circular_layout(graph)
    pos = {
        node: (
            float(base_pos[node][0] + 0.24 * rng.normal()),
            float(base_pos[node][1] + 0.24 * rng.normal()),
        )
        for node in graph.nodes()
    }
    # Ensure no two customer nodes end up visually overlapping.
    min_dist = 0.20
    for node_a in graph.nodes():
        for node_b in graph.nodes():
            if node_a >= node_b:
                continue
            xa, ya = pos[node_a]
            xb, yb = pos[node_b]
            dx, dy = xa - xb, ya - yb
            dist = float(np.hypot(dx, dy))
            if dist < min_dist:
                if dist < 1e-6:
                    dx, dy = 1.0, 0.0
                    dist = 1.0
                ux, uy = dx / dist, dy / dist
                shift = 0.5 * (min_dist - dist)
                pos[node_a] = (xa + ux * shift, ya + uy * shift)
                pos[node_b] = (xb - ux * shift, yb - uy * shift)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    nx.draw_networkx_edges(graph, pos, width=2.8, alpha=0.9, edge_color="#2f2f2f", ax=ax)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color="#6d8fb3",
        edgecolors="#1f1f1f",
        node_size=760,
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=9, font_weight="bold", ax=ax)

    ax.set_title("Customer interaction graph")
    ax.axis("off")
    fig.savefig(OUTPUT_DIR / "interaction_graph_compact.png", dpi=220)
    plt.close(fig)


def plot_relational_dataset_graph(customers: pd.DataFrame, transactions: pd.DataFrame, interactions: pd.DataFrame) -> None:
    graph = nx.Graph()
    positions: dict[str, tuple[float, float]] = {}
    customer_ids = customers["customer_id"].tolist()
    customer_y = {customer_id: float(len(customer_ids) - idx - 1) for idx, customer_id in enumerate(customer_ids)}

    for customer_id in customer_ids:
        graph.add_node(customer_id, node_type="customer")
        positions[customer_id] = (0.0, customer_y[customer_id])

    # Arrange transaction-row nodes in two slim columns to the left of their customer.
    transaction_offsets = [0.45, 0.25, 0.08, -0.08, -0.25, -0.45]
    transaction_x = [-1.3, -1.05, -1.3, -1.05, -1.3, -1.05]
    for customer_id, customer_transactions in transactions.groupby("customer_id", sort=False):
        for local_idx, row in enumerate(customer_transactions.itertuples(index=False), start=1):
            txn_node = f"txn_{customer_id}_{local_idx}"
            graph.add_node(txn_node, node_type="transaction")
            graph.add_edge(txn_node, customer_id)
            positions[txn_node] = (
                transaction_x[local_idx - 1],
                customer_y[customer_id] + transaction_offsets[local_idx - 1],
            )

    # Arrange interaction-row nodes to the right, near the midpoint of the two linked customers.
    right_x_cycle = [1.05, 1.25, 1.45]
    right_y_cycle = [0.18, 0.0, -0.18]
    for idx, row in enumerate(interactions.itertuples(index=False), start=1):
        interaction_node = f"int_{idx:02d}"
        graph.add_node(interaction_node, node_type="interaction")
        graph.add_edge(interaction_node, row.customer_i)
        graph.add_edge(interaction_node, row.customer_j)
        midpoint_y = (customer_y[row.customer_i] + customer_y[row.customer_j]) / 2.0
        positions[interaction_node] = (
            right_x_cycle[(idx - 1) % len(right_x_cycle)],
            midpoint_y + right_y_cycle[(idx - 1) % len(right_y_cycle)],
        )

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    nx.draw_networkx_edges(graph, positions, width=0.7, alpha=0.18, edge_color="#777777", ax=ax)

    type_style = {
        "customer": {"color": "#4c78a8", "size": 460, "linewidth": 1.0},
        "transaction": {"color": "#59a14f", "size": 36, "linewidth": 0.4},
        "interaction": {"color": "#f28e2b", "size": 52, "linewidth": 0.4},
    }
    for node_type, style in type_style.items():
        node_list = [node for node, attrs in graph.nodes(data=True) if attrs["node_type"] == node_type]
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=node_list,
            node_color=style["color"],
            edgecolors="black",
            node_size=style["size"],
            linewidths=style["linewidth"],
            alpha=0.92,
            ax=ax,
        )

    customer_labels = {customer_id: customer_id for customer_id in customer_ids}
    nx.draw_networkx_labels(graph, positions, labels=customer_labels, font_size=9, font_weight="bold", ax=ax)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c78a8", markeredgecolor="black", markersize=9, label="Customer rows"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#59a14f", markeredgecolor="black", markersize=6, label="Transaction rows"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f28e2b", markeredgecolor="black", markersize=7, label="Interaction rows"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False)
    ax.set_title(
        "Row-to-node graph view of the relational dataset\n"
        "Blue = customer rows, green = transaction rows, orange = interaction rows"
    )
    ax.set_axis_off()
    fig.savefig(OUTPUT_DIR / "relational_dataset_graph.png", dpi=220)
    plt.close(fig)


def plot_source_tables(customers: pd.DataFrame, transactions: pd.DataFrame, interactions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    table_specs = [
        (
            axes[0],
            customers[["customer_id", "age", "income_k", "tenure_months", "region", "channel_pref"]].head(6),
            "Customers table\n(first 6 rows)",
            [0.18, 0.1, 0.14, 0.18, 0.16, 0.2],
        ),
        (
            axes[1],
            transactions[["customer_id", "month", "category", "amount"]].head(8),
            "Transactions table\n(first 8 rows)",
            [0.2, 0.12, 0.34, 0.18],
        ),
        (
            axes[2],
            interactions.head(8),
            "Interactions table\n(first 8 rows)",
            [0.32, 0.32],
        ),
    ]
    for ax, frame, title, widths in table_specs:
        ax.axis("off")
        rendered = ax.table(
            cellText=frame.astype(str).values,
            colLabels=frame.columns.tolist(),
            cellLoc="center",
            loc="center",
            colWidths=widths,
        )
        rendered.auto_set_font_size(False)
        rendered.set_fontsize(8)
        rendered.scale(1.0, 1.35)
        for (row, col), cell in rendered.get_celld().items():
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor("#e9eef5")
                cell.set_text_props(weight="bold")
        ax.set_title(title, fontsize=12)
    fig.suptitle("Running example: three relational tables", fontsize=14)
    fig.savefig(OUTPUT_DIR / "source_tables_overview.png", dpi=220)
    plt.close(fig)


def plot_feature_heatmap(feature_df: pd.DataFrame) -> None:
    ordered = feature_df.sort_values(["feature_segment", "customer_id"]).reset_index(drop=True)
    numeric_cols = [
        "age",
        "income_k",
        "tenure_months",
        "total_spend",
        "avg_transaction",
        "spend_electronics",
        "spend_fashion",
        "spend_groceries",
        "spend_home",
    ]
    standardized = zscore_frame(ordered[numeric_cols]).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    image = ax.imshow(standardized, aspect="auto", cmap="coolwarm", vmin=-2.2, vmax=2.2)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=35, ha="right")
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(
        [f"{cid} ({seg})" for cid, seg in zip(ordered["customer_id"], ordered["feature_segment"])],
        fontsize=8,
    )
    ax.set_title("Standardized feature matrix X\nRows ordered by feature segment")
    ax.set_xlabel("Feature columns")
    ax.set_ylabel("Customers")
    for boundary in [3.5, 7.5]:
        ax.axhline(boundary, color="black", linewidth=1.0)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.92)
    colorbar.set_label("Standardized value")
    fig.savefig(OUTPUT_DIR / "feature_matrix_heatmap.png", dpi=220)
    plt.close(fig)


def plot_segment_profiles(feature_df: pd.DataFrame) -> None:
    spend_cols = ["spend_electronics", "spend_fashion", "spend_groceries", "spend_home"]
    profile = feature_df.groupby("feature_segment")[spend_cols].mean().loc[["budget", "premium", "family"]]
    x_positions = np.arange(len(spend_cols))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    for idx, segment in enumerate(profile.index):
        ax.bar(
            x_positions + (idx - 1) * width,
            profile.loc[segment].to_numpy(),
            width=width,
            color=SEGMENT_COLORS[segment],
            label=segment,
            edgecolor="black",
            linewidth=0.7,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["electronics", "fashion", "groceries", "home"])
    ax.set_ylabel("Average spend")
    ax.set_title("Average category spend by feature segment")
    ax.legend(title="Segment", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT_DIR / "segment_spend_profiles.png", dpi=220)
    plt.close(fig)


def plot_singular_values(singular_values: np.ndarray) -> None:
    energy = singular_values ** 2
    cumulative = np.cumsum(energy) / np.sum(energy)
    components = np.arange(1, len(singular_values) + 1)

    fig, ax1 = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax1.bar(components, energy / np.sum(energy), color="#4c78a8", edgecolor="black", linewidth=0.7)
    ax1.set_xlabel("Component index")
    ax1.set_ylabel("Explained energy ratio")
    ax1.set_xticks(components)
    ax1.set_title("Truncated SVD diagnostic: singular-value energy")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(components, cumulative, color="#d62728", marker="o", linewidth=2)
    ax2.set_ylabel("Cumulative energy")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(cumulative[1], color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)
    ax2.text(2.15, cumulative[1] + 0.015, f"Top 2 = {cumulative[1]:.3f}", color="#d62728", fontsize=9)
    fig.savefig(OUTPUT_DIR / "svd_singular_values.png", dpi=220)
    plt.close(fig)


def plot_graph_matrices(customers: pd.DataFrame, adjacency: np.ndarray, laplacian: np.ndarray) -> None:
    ordered = customers.sort_values(["social_circle", "customer_id"]).reset_index(drop=True)
    ordered_ids = ordered["customer_id"].tolist()
    id_to_idx = {customer_id: idx for idx, customer_id in enumerate(customers["customer_id"])}
    order = np.array([id_to_idx[customer_id] for customer_id in ordered_ids])
    adjacency_ordered = adjacency[order][:, order]
    laplacian_ordered = laplacian[order][:, order]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    image_a = axes[0].imshow(adjacency_ordered, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("Adjacency matrix A\nCustomers ordered by social circle")
    image_l = axes[1].imshow(laplacian_ordered, cmap="coolwarm")
    axes[1].set_title("Graph Laplacian L = D - A")
    for ax in axes:
        ax.set_xticks(range(len(ordered_ids)))
        ax.set_xticklabels(ordered_ids, rotation=90, fontsize=8)
        ax.set_yticks(range(len(ordered_ids)))
        ax.set_yticklabels(ordered_ids, fontsize=8)
        ax.axhline(5.5, color="black", linewidth=1.0)
        ax.axvline(5.5, color="black", linewidth=1.0)
    fig.colorbar(image_a, ax=axes[0], shrink=0.86)
    fig.colorbar(image_l, ax=axes[1], shrink=0.86)
    fig.savefig(OUTPUT_DIR / "adjacency_laplacian_matrices.png", dpi=220)
    plt.close(fig)


def plot_laplacian_spectrum(eigenvalues: np.ndarray) -> None:
    indices = np.arange(len(eigenvalues))
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.plot(indices, eigenvalues, marker="o", color="#4c78a8", linewidth=1.8)
    ax.scatter(indices[:3], eigenvalues[:3], color="#d62728", zorder=3)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Laplacian spectrum")
    ax.grid(alpha=0.25)
    ax.annotate(r"$\lambda_0 \approx 0$", (0, eigenvalues[0]), xytext=(0.7, eigenvalues[2] * 0.25),
                arrowprops=dict(arrowstyle="->", linewidth=1.0), fontsize=9)
    ax.annotate(r"$\lambda_1$ (first nontrivial)", (1, eigenvalues[1]), xytext=(2.2, eigenvalues[2] * 0.55),
                arrowprops=dict(arrowstyle="->", linewidth=1.0), fontsize=9)
    fig.savefig(OUTPUT_DIR / "laplacian_spectrum.png", dpi=220)
    plt.close(fig)


def plot_fiedler_vector(customers: pd.DataFrame, eigenvectors: np.ndarray) -> None:
    ordered = customers.sort_values(["social_circle", "customer_id"]).reset_index(drop=True)
    id_to_idx = {customer_id: idx for idx, customer_id in enumerate(customers["customer_id"])}
    values = []
    for customer_id in ordered["customer_id"]:
        values.append(eigenvectors[id_to_idx[customer_id], 1])

    colors = [COMMUNITY_COLORS[circle] for circle in ordered["social_circle"]]
    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    ax.bar(ordered["customer_id"], values, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Customers ordered by social circle")
    ax.set_ylabel("Value in second eigenvector")
    ax.set_title("Fiedler vector values by customer")
    ax.grid(axis="y", alpha=0.25)
    north_values = values[:6]
    south_values = values[6:]
    ax.text(2.2, min(north_values) - 0.05, "north_circle", color=COMMUNITY_COLORS["north_circle"], fontsize=10)
    ax.text(7.0, max(south_values) + 0.04, "south_circle", color=COMMUNITY_COLORS["south_circle"], fontsize=10)
    fig.savefig(OUTPUT_DIR / "fiedler_vector_by_customer.png", dpi=220)
    plt.close(fig)


def write_report(
    feature_df: pd.DataFrame,
    svd_df: pd.DataFrame,
    lap_df: pd.DataFrame,
    svd_ratio: np.ndarray,
    eigenvalues: np.ndarray,
    adjacency: np.ndarray,
) -> None:
    customer_summary = feature_df[["customer_id", "feature_segment", "social_circle"]]
    svd_segment_score = separation_score(svd_df, "svd_x", "svd_y", "feature_segment")
    svd_circle_score = separation_score(svd_df, "svd_x", "svd_y", "social_circle")
    lap_circle_score = separation_score(lap_df, "laplacian_x", "laplacian_y", "social_circle")
    lap_segment_score = separation_score(lap_df, "laplacian_x", "laplacian_y", "feature_segment")

    with open(OUTPUT_DIR / "example_report.txt", "w", encoding="utf-8") as handle:
        handle.write("Customer DR example summary\n")
        handle.write("===========================\n\n")
        handle.write("Design intent:\n")
        handle.write("- Truncated SVD should group customers by purchase behavior.\n")
        handle.write("- Laplacian Eigenmaps should group customers by interaction-circle structure.\n\n")
        handle.write("Customer labels:\n")
        handle.write(customer_summary.to_string(index=False))
        handle.write("\n\n")
        handle.write(f"Top-2 singular-value energy ratio: {svd_ratio.sum():.4f}\n")
        handle.write(f"Individual SVD energy ratios: {svd_ratio[0]:.4f}, {svd_ratio[1]:.4f}\n")
        handle.write("Smallest Laplacian eigenvalues:\n")
        handle.write(", ".join(f"{value:.4f}" for value in eigenvalues[:6]))
        handle.write("\n")
        handle.write(f"Graph edges: {int(adjacency.sum() / 2)}\n")
        handle.write(f"Graph connected: {'yes' if eigenvalues[1] > 1e-8 else 'no'}\n")
        handle.write("\nSeparation checks (higher is better):\n")
        handle.write(f"- SVD grouped by feature_segment: {svd_segment_score:.4f}\n")
        handle.write(f"- SVD grouped by social_circle:  {svd_circle_score:.4f}\n")
        handle.write(f"- Laplacian grouped by social_circle:  {lap_circle_score:.4f}\n")
        handle.write(f"- Laplacian grouped by feature_segment: {lap_segment_score:.4f}\n")

        if svd_segment_score > svd_circle_score and lap_circle_score > lap_segment_score:
            handle.write("\nValidation: PASS\n")
            handle.write("The example separates the intended notion of similarity in each method.\n")
        else:
            handle.write("\nValidation: CHECK MANUALLY\n")
            handle.write("One of the two methods is not emphasizing the intended grouping strongly enough.\n")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    customers = build_customers()
    transactions = build_transactions(customers)
    interactions = build_interactions()
    feature_df = build_feature_matrix(customers, transactions)

    svd_coords, svd_ratio, singular_values = truncated_svd_embedding(feature_df)
    lap_coords, adjacency, laplacian, eigenvalues, eigenvectors, _ = laplacian_eigenmaps_embedding(customers, interactions)

    save_tables(customers, transactions, interactions, feature_df)
    write_data_description()

    svd_df = make_embedding_df(feature_df, svd_coords, "svd")
    lap_df = make_embedding_df(feature_df, lap_coords, "laplacian")
    svd_df.to_csv(OUTPUT_DIR / "svd_embedding.csv", index=False)
    lap_df.to_csv(OUTPUT_DIR / "laplacian_embedding.csv", index=False)

    plot_source_tables(customers, transactions, interactions)
    plot_feature_heatmap(feature_df)
    plot_segment_profiles(feature_df)
    plot_singular_values(singular_values)
    plot_graph_matrices(customers, adjacency, laplacian)
    plot_laplacian_spectrum(eigenvalues)
    plot_fiedler_vector(customers, eigenvectors)
    plot_embeddings(svd_df, lap_df)
    plot_graph(customers, interactions)
    plot_compact_relation_graph(customers, interactions)
    plot_relational_dataset_graph(customers, transactions, interactions)
    write_report(feature_df, svd_df, lap_df, svd_ratio, eigenvalues, adjacency)

    print("Saved data files:")
    for path in sorted(DATA_DIR.glob("*.csv")):
        print(f" - {path.relative_to(ROOT)}")
    print("Saved output files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f" - {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
