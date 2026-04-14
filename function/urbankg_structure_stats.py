"""
Compute structural statistics for an UrbanKG from Neo4j using .env config.

Outputs:
- total entities / relations / triples
- entity type counts
- relation type counts
- triple type counts
- graph connected components
- cycle count (independent cycles / cyclomatic number on undirected simple graph)
- approximate graph hyperbolicity on the largest connected component

Notes:
- "cycle" follows a tractable graph-structural proxy: m - n + c on the undirected
  simple graph, i.e. the number of independent cycles.
- "hyperbolicity" is approximated using sampled quadruples and the Gromov
  four-point condition on shortest-path distances.
"""

import argparse
import itertools
import json
import math
import os
import random
from collections import Counter
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
import networkx as nx


def gromov_delta_from_distances(dab: int, dac: int, dad: int, dbc: int, dbd: int, dcd: int) -> float:
    sums = sorted([
        dab + dcd,
        dac + dbd,
        dad + dbc,
    ])
    return (sums[2] - sums[1]) / 2.0


def approximate_hyperbolicity(
    graph: nx.Graph,
    sample_nodes: int = 96,
    sample_quadruples: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    if graph.number_of_nodes() < 4 or graph.number_of_edges() == 0:
        return {
            "largest_component_nodes": graph.number_of_nodes(),
            "sampled_nodes": graph.number_of_nodes(),
            "sampled_quadruples": 0,
            "approx_hyperbolicity": 0.0,
        }

    rng = random.Random(seed)
    components = list(nx.connected_components(graph))
    largest_nodes = max(components, key=len)
    gcc = graph.subgraph(largest_nodes).copy()

    nodes = list(gcc.nodes())
    sampled = nodes if len(nodes) <= sample_nodes else rng.sample(nodes, sample_nodes)

    dists: Dict[int, Dict[int, int]] = {}
    for node in sampled:
        dd = nx.single_source_shortest_path_length(gcc, node)
        dists[node] = {k: v for k, v in dd.items() if k in sampled}

    quadruples = list(itertools.combinations(sampled, 4))
    if not quadruples:
        return {
            "largest_component_nodes": gcc.number_of_nodes(),
            "sampled_nodes": len(sampled),
            "sampled_quadruples": 0,
            "approx_hyperbolicity": 0.0,
        }
    if len(quadruples) > sample_quadruples:
        quadruples = rng.sample(quadruples, sample_quadruples)

    max_delta = 0.0
    valid = 0
    for a, b, c, d in quadruples:
        try:
            delta = gromov_delta_from_distances(
                dists[a][b], dists[a][c], dists[a][d],
                dists[b][c], dists[b][d], dists[c][d],
            )
        except KeyError:
            continue
        max_delta = max(max_delta, delta)
        valid += 1

    return {
        "largest_component_nodes": gcc.number_of_nodes(),
        "sampled_nodes": len(sampled),
        "sampled_quadruples": valid,
        "approx_hyperbolicity": max_delta if valid else 0.0,
    }


def fetch_basic_stats(session) -> Dict[str, object]:
    label_rows = session.run(
        "CALL db.labels() YIELD label RETURN label ORDER BY label"
    ).data()
    rel_rows = session.run(
        "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
    ).data()
    entity_count_rows = session.run(
        """
        MATCH (n)
        UNWIND labels(n) AS label
        RETURN label, count(*) AS count
        ORDER BY count DESC, label
        """
    ).data()
    relationship_count_rows = session.run(
        """
        MATCH ()-[r]->()
        RETURN type(r) AS relationshipType, count(*) AS count
        ORDER BY count DESC, relationshipType
        """
    ).data()
    triple_type_rows = session.run(
        """
        MATCH (s)-[r]->(o)
        UNWIND labels(s) AS s_label
        UNWIND labels(o) AS o_label
        RETURN s_label AS start_label,
               type(r) AS relationship_type,
               o_label AS end_label,
               count(*) AS count
        ORDER BY count DESC, start_label, relationship_type, end_label
        """
    ).data()
    totals = session.run(
        """
        MATCH (n)
        WITH count(n) AS total_entities
        MATCH ()-[r]->()
        RETURN total_entities, count(r) AS total_relationships
        """
    ).single()
    return {
        "entity_types": [row["label"] for row in label_rows],
        "relationship_types": [row["relationshipType"] for row in rel_rows],
        "entity_counts": {row["label"]: row["count"] for row in entity_count_rows},
        "relationship_counts": {row["relationshipType"]: row["count"] for row in relationship_count_rows},
        "triple_types": [
            {
                "start_label": row["start_label"],
                "relationship_type": row["relationship_type"],
                "end_label": row["end_label"],
                "count": row["count"],
            }
            for row in triple_type_rows
        ],
        "total_entities": totals["total_entities"] if totals else 0,
        "total_relationships": totals["total_relationships"] if totals else 0,
        "total_triples": totals["total_relationships"] if totals else 0,
    }


def build_undirected_graph(session) -> nx.Graph:
    graph = nx.Graph()
    result = session.run("MATCH (s)-[r]->(o) RETURN id(s) AS s, id(o) AS o")
    for row in result:
        s = row["s"]
        o = row["o"]
        if s is None or o is None:
            continue
        if s == o:
            graph.add_node(s)
            continue
        graph.add_edge(s, o)
    return graph


def compute_graph_structure(graph: nx.Graph) -> Dict[str, object]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    components = list(nx.connected_components(graph))
    c = len(components)
    cycle_count = max(0, m - n + c)
    component_sizes = sorted((len(comp) for comp in components), reverse=True)
    return {
        "graph_nodes_undirected": n,
        "graph_edges_undirected": m,
        "connected_components": c,
        "largest_component_size": component_sizes[0] if component_sizes else 0,
        "cycle_count": cycle_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute UrbanKG structural statistics from Neo4j")
    parser.add_argument("--database", default=None, help="Optional Neo4j database override; defaults to .env NEO4J_DATABASE")
    parser.add_argument("--sample-nodes", type=int, default=96, help="Number of sampled nodes for hyperbolicity approximation")
    parser.add_argument("--sample-quadruples", type=int, default=2000, help="Number of sampled quadruples for hyperbolicity approximation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable text")
    args = parser.parse_args()

    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    db = args.database or os.getenv("NEO4J_DATABASE")

    if not uri or not user or not pwd:
        raise RuntimeError("NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD not set in .env")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        with driver.session(database=db) as session:
            basic = fetch_basic_stats(session)
            graph = build_undirected_graph(session)

        structure = compute_graph_structure(graph)
        hyper = approximate_hyperbolicity(
            graph,
            sample_nodes=args.sample_nodes,
            sample_quadruples=args.sample_quadruples,
            seed=args.seed,
        )

        output = {
            "database": db,
            "entity_total": basic["total_entities"],
            "relation_total": basic["total_relationships"],
            "triplet_total": basic["total_triples"],
            "entity_type_total": len(basic["entity_types"]),
            "relation_type_total": len(basic["relationship_types"]),
            "cycle_count": structure["cycle_count"],
            "approx_hyperbolicity": hyper["approx_hyperbolicity"],
            "graph_structure": structure,
            "hyperbolicity_detail": hyper,
            "entity_counts": basic["entity_counts"],
            "relation_counts": basic["relationship_counts"],
            "triple_types": basic["triple_types"],
        }

        if args.json:
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return

        print(f"Database: {db}")
        print("\nTotals:")
        print(f"- Entities: {output['entity_total']}")
        print(f"- Relations: {output['relation_total']}")
        print(f"- Triplets: {output['triplet_total']}")
        print(f"- Entity types: {output['entity_type_total']}")
        print(f"- Relation types: {output['relation_type_total']}")

        print("\nStructural indicators:")
        print(f"- Cycle count: {output['cycle_count']}")
        print(f"- Approx. hyperbolicity: {output['approx_hyperbolicity']:.4f}")
        print(f"- Connected components: {structure['connected_components']}")
        print(f"- Largest component size: {structure['largest_component_size']}")
        print(f"- Undirected graph nodes: {structure['graph_nodes_undirected']}")
        print(f"- Undirected graph edges: {structure['graph_edges_undirected']}")
        print(f"- Hyperbolicity sampled nodes: {hyper['sampled_nodes']}")
        print(f"- Hyperbolicity sampled quadruples: {hyper['sampled_quadruples']}")

        print("\nEntity counts by type:")
        for label, count in basic["entity_counts"].items():
            print(f"- {label}: {count}")

        print("\nRelation counts by type:")
        for rel, count in basic["relationship_counts"].items():
            print(f"- {rel}: {count}")

        print("\nTriple types:")
        for row in basic["triple_types"]:
            print(f"- ({row['start_label']})-[:{row['relationship_type']}]->({row['end_label']}): {row['count']}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
