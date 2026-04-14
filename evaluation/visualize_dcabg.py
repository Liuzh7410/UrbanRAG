"""
Visualize DCABG inputs/outputs for a selected address.

Plots:
- Area boundary
- Roads
- Seed points (POI/Block/Area centroid)
- DCABG output polygon
"""

import argparse
import csv
import os
from typing import List, Optional

# Ensure OpenAI client init in main.py does not fail without a key
os.environ.setdefault("OPENAI_API_KEY", "DUMMY")

import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, Polygon, Point

import main as kg


def load_addresses(csv_path: str, column: str) -> List[str]:
    addresses = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if column not in fieldnames:
            column = fieldnames[0] if fieldnames else column
        for row in reader:
            value = row.get(column)
            if value:
                addresses.append(value.strip())
    return addresses


def select_address(addresses: List[str]) -> Optional[str]:
    for address in addresses:
        parsed = kg.parse_address(address)
        if not parsed:
            continue
        indices = kg.build_indices(parsed)
        block_info, sibling_pois = kg.backtracking_retrieval(indices["Block"])
        seeds, seed_metadata = kg.determine_seeds(block_info, sibling_pois, parsed, indices)
        if not seeds:
            area_polygon, _ = kg.query_area_and_roads(indices["Area"])
            if area_polygon:
                centroid = area_polygon.centroid
                seeds = [Point(centroid.x, centroid.y)]
                seed_metadata = {"case": "C", "seed_count": 1, "source": "area_centroid"}
        if not seeds:
            continue
        spatial_constraint = kg.dcabg_algorithm(indices, seeds, parsed, seed_metadata)
        if spatial_constraint:
            return address
    return None


def plot_geometry(ax, geom, **kwargs):
    if isinstance(geom, LineString):
        x, y = geom.xy
        ax.plot(x, y, **kwargs)
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            x, y = line.xy
            ax.plot(x, y, **kwargs)
    elif isinstance(geom, Polygon):
        x, y = geom.exterior.xy
        ax.plot(x, y, **kwargs)
    elif isinstance(geom, Point):
        ax.scatter([geom.x], [geom.y], **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Visualize DCABG algorithm result")
    parser.add_argument("--address", default=None, help="Address to visualize")
    parser.add_argument("--input", default="data_sample/shizuoka_city_addresses.csv")
    parser.add_argument("--column", default="place_name")
    parser.add_argument("--output", default="results/visualization/dcabg_example.png")
    args = parser.parse_args()

    if args.address:
        address = args.address.strip()
    else:
        addresses = load_addresses(args.input, args.column)
        address = select_address(addresses)
        if not address:
            raise RuntimeError("No suitable address found for DCABG visualization")

    address = kg.normalize_address(address)
    parsed = kg.parse_address(address)
    if not parsed:
        raise RuntimeError(f"Address parse failed: {address}")

    indices = kg.build_indices(parsed)
    block_info, sibling_pois = kg.backtracking_retrieval(indices["Block"])
    seeds, seed_metadata = kg.determine_seeds(block_info, sibling_pois, parsed, indices)
    if not seeds:
        area_polygon, _ = kg.query_area_and_roads(indices["Area"])
        if not area_polygon:
            raise RuntimeError("No seeds available for DCABG visualization")
        centroid = area_polygon.centroid
        seeds = [Point(centroid.x, centroid.y)]
        seed_metadata = {"case": "C", "seed_count": 1, "source": "area_centroid"}

    spatial_constraint = kg.dcabg_algorithm(indices, seeds, parsed, seed_metadata)
    if not spatial_constraint:
        raise RuntimeError("DCABG failed to produce a polygon")

    area_polygon, roads = kg.query_area_and_roads(indices["Area"])
    if not area_polygon:
        raise RuntimeError("Area geometry not found")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Area boundary
    plot_geometry(ax, area_polygon, color="black", linewidth=1.5, label="Area boundary")

    # Roads
    for road in roads:
        plot_geometry(ax, road, color="gray", linewidth=0.8, alpha=0.7)

    # Seeds
    for seed in seeds:
        plot_geometry(ax, seed, color="red", s=20, label="Seed" if seed == seeds[0] else None)

    # DCABG polygon
    dcabg_polygon = spatial_constraint["polygon"]
    plot_geometry(ax, dcabg_polygon, color="blue", linewidth=2.0, label="DCABG polygon")

    ax.set_title(f"DCABG Visualization\n{address}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Address: {address}")
    print(f"Seeds: {seed_metadata.get('seed_count')} | Case: {seed_metadata.get('case')}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
