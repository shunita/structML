from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def build_relational_graph(
    customers: pd.DataFrame, transactions: pd.DataFrame, interactions: pd.DataFrame
) -> nx.Graph:
    """Convert relational rows into a heterogeneous graph."""
    graph = nx.Graph()

    # 1) customer rows -> customer nodes
    for row in customers.itertuples(index=False):
        graph.add_node(
            row.customer_id,
            node_type="customer",
            region=row.region,
            channel_pref=row.channel_pref,
        )

    # 2) transaction rows -> transaction nodes linked to their owning customer
    for idx, row in enumerate(transactions.itertuples(index=False), start=1):
        txn_node = f"txn_{idx:03d}"
        graph.add_node(txn_node, node_type="transaction", category=row.category, amount=row.amount)
        graph.add_edge(txn_node, row.customer_id, relation="made_transaction")

    # 3) interaction rows -> interaction nodes linked to both customers
    for idx, row in enumerate(interactions.itertuples(index=False), start=1):
        int_node = f"int_{idx:03d}"
        graph.add_node(int_node, node_type="interaction")
        graph.add_edge(int_node, row.customer_i, relation="interaction_endpoint")
        graph.add_edge(int_node, row.customer_j, relation="interaction_endpoint")

    return graph


def run_demo() -> None:
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    interactions = pd.read_csv(DATA_DIR / "interactions.csv")

    graph = build_relational_graph(customers, transactions, interactions)

    expected_nodes = len(customers) + len(transactions) + len(interactions)
    expected_edges = len(transactions) + 2 * len(interactions)

    assert graph.number_of_nodes() == expected_nodes, "Unexpected node count"
    assert graph.number_of_edges() == expected_edges, "Unexpected edge count"

    print(f"nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")
    print(f"node types={dict(pd.Series(nx.get_node_attributes(graph, 'node_type')).value_counts())}")


if __name__ == "__main__":
    run_demo()
