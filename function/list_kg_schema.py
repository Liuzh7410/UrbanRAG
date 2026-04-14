"""
List schema and size statistics from Neo4j using .env config.

Outputs:
- entity types
- relationship types
- triple types
- entity counts
- relationship counts
- triple counts
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase


def main():
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE")

    if not uri or not user or not pwd:
        raise RuntimeError("NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD not set in .env")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        with driver.session(database=db) as session:
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
            total_entity_row = session.run(
                "MATCH (n) RETURN count(n) AS total_entities"
            ).single()
            total_relationship_row = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS total_relationships"
            ).single()

        total_triples = total_relationship_row["total_relationships"] if total_relationship_row else 0
        total_entities = total_entity_row["total_entities"] if total_entity_row else 0
        total_relationships = total_relationship_row["total_relationships"] if total_relationship_row else 0

        print("Entity types:")
        for row in label_rows:
            print(f"- {row['label']}")

        print("\nRelationship types:")
        for row in rel_rows:
            print(f"- {row['relationshipType']}")

        print("\nTriple types:")
        for row in triple_type_rows:
            print(
                f"- ({row['start_label']})-[:{row['relationship_type']}]->({row['end_label']}): {row['count']}"
            )

        print("\nEntity counts by type:")
        for row in entity_count_rows:
            print(f"- {row['label']}: {row['count']}")

        print("\nRelationship counts by type:")
        for row in relationship_count_rows:
            print(f"- {row['relationshipType']}: {row['count']}")

        print("\nTotals:")
        print(f"- Total entities: {total_entities}")
        print(f"- Total relationships: {total_relationships}")
        print(f"- Total triples: {total_triples}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
