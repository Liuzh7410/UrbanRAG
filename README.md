# UrbanKG_GraphRAG

UrbanKG_GraphRAG is a knowledge-graph–enhanced geocoding research system that combines Urban Knowledge Graph retrieval with spatial constraint reasoning. It targets Japanese addresses and improves coordinate inference by grounding LLM reasoning in graph-derived spatial context.

## Research Summary

The system integrates three key components:

1. **Urban Knowledge Graph (Neo4j)**
   - Entities: POI, Block, Area, Road, Ward
   - Relations: `locatesAt`, `belongsTo`, `in`, `nearby`, `borderBy`

2. **Backtracking Retrieval**
   - Exact POI match → sibling POIs in same Block
   - If Block missing, neighbor Blocks are searched (limited radius)
   - If still missing, Area-level POIs are used as fallback

3. **Spatial Constraint Algorithm (DCABG)**
   - Uses Area boundary + Road network to generate a virtual block polygon
   - Seeds (POIs/Block point/Area centroid) anchor the correct polygon
   - Outputs centroid/bounds/radius to constrain LLM reasoning

## Workflow (High-Level)

1. Parse address and build indices (Ward / Area / Block)
2. Exact POI retrieval (one-to-one / one-to-many)
3. Backtracking retrieval for one-to-zero
4. DCABG spatial constraint generation
5. LLM reasoning within constraints
6. Evaluation vs Google Maps API

## Baselines and Experiments

- **LLM-only** (no KG, no constraints)
- **Plain Text RAG** (retrieval over KG text corpus)
- **LightRAG (Textural KG)** (retrieval over node/edge textualized corpus)
- **GraphRAG (UrbanKG + DCABG)** (main method)

Experiments are conducted with 100 / 1000 / 10000 samples for each dataset.

## Datasets

- Tokyo, Shizuoka, Susono
- Input addresses in `data_sample/`
- Neo4j export and corpus files in `corpus/`

## Outputs

- Geocoding results → `results/`
- Evaluation outputs → `results/` and `results/reports/`
- Visualization outputs → `results/visualization/` and `visualization/`

---

This repository is research-oriented and focuses on reproducible geocoding experiments with knowledge-graph grounding and spatial constraints.
