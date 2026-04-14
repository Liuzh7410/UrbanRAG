"""
Visualize one-to-zero cases with Google (red), model (blue), and Neo4j (black) points.
Case C also overlays Area polygon.
"""

import os
import csv
import json
import re
from typing import List, Tuple, Optional

import folium
from shapely import wkt


def extract_wkt_from_neo4j_string(value: str) -> Optional[str]:
    if not value:
        return None
    m = re.search(r"(POINT\\s*\\([^\\)]*\\)|POLYGON\\s*\\(\\([^\\)]*\\)\\))", value)
    if not m:
        return None
    return m.group(1)


def load_points_from_json(path: str) -> List[Tuple[float, float]]:
    points = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return points
    for row in data:
        p = row.get("p", {})
        props = p.get("properties", {})
        geom = props.get("geometry")
        if not geom:
            continue
        try:
            g = wkt.loads(geom)
            if g.geom_type == "Point":
                points.append((g.y, g.x))
        except Exception:
            continue
    return points


def load_points_from_csv(path: str) -> List[Tuple[float, float]]:
    points = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("p")
            if not raw:
                continue
            try:
                geom = extract_wkt_from_neo4j_string(raw) or raw
                g = wkt.loads(geom)
                if g.geom_type == "Point":
                    points.append((g.y, g.x))
            except Exception:
                continue
    return points


def load_polygon_from_json(path: str) -> Optional[List[Tuple[float, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return None
    for row in data:
        p = row.get("p", {})
        props = p.get("properties", {})
        geom = props.get("geometry")
        if not geom:
            continue
        try:
            g = wkt.loads(geom)
            if g.geom_type == "Polygon":
                return [(lat, lon) for lon, lat in list(g.exterior.coords)]
        except Exception:
            continue
    return None


def load_polygon_from_csv(path: str) -> Optional[List[Tuple[float, float]]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("p")
            if not raw:
                continue
            try:
                geom = extract_wkt_from_neo4j_string(raw) or raw
                g = wkt.loads(geom)
                if g.geom_type == "Polygon":
                    return [(lat, lon) for lon, lat in list(g.exterior.coords)]
            except Exception:
                continue
    return None


def add_point(fmap, lat: float, lon: float, color: str, label: str, radius: int = 6):
    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        popup=label
    ).add_to(fmap)


def render_case(output_path: str,
                google: Tuple[float, float],
                model: Tuple[float, float],
                neo4j_points: List[Tuple[float, float]],
                polygon: Optional[List[Tuple[float, float]]] = None):
    fmap = folium.Map(location=[google[0], google[1]], zoom_start=18, tiles="CartoDB positron")

    add_point(fmap, google[0], google[1], "red", "Google Maps", radius=6)
    add_point(fmap, model[0], model[1], "blue", "Model", radius=6)

    print(f"Neo4j points: {len(neo4j_points)} -> {output_path}")
    for lat, lon in neo4j_points:
        add_point(fmap, lat, lon, "black", "Neo4j POI", radius=4)

    if polygon:
        folium.Polygon(
            locations=polygon,
            color="lightblue",
            weight=8,
            fill=False,
            popup="Area"
        ).add_to(fmap)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fmap.save(output_path)


def main():
    # Case A
    case_a_csv = "data_sample/neo4j_csv/敷地 1丁目 7.csv"
    case_a_json = "data_sample/neo4j_csv/Case_a.json"
    case_a_google = (34.951173, 138.4104075)
    case_a_model = (34.951217, 138.410296)
    case_a_points = load_points_from_json(case_a_json)
    if not case_a_points:
        case_a_points = load_points_from_csv(case_a_csv)
    render_case(
        output_path="results/visualization/one_to_zero_case_a.html",
        google=case_a_google,
        model=case_a_model,
        neo4j_points=case_a_points
    )

    # Case B
    case_b_csv = "data_sample/neo4j_csv/大岩 1丁目.csv"
    case_b_json = "data_sample/neo4j_csv/Case_b.json"
    case_b_google = (34.9925154, 138.3818482)
    case_b_model = (34.992860, 138.381692)
    case_b_points = load_points_from_json(case_b_json)
    if not case_b_points:
        case_b_points = load_points_from_csv(case_b_csv)
    render_case(
        output_path="results/visualization/one_to_zero_case_b.html",
        google=case_b_google,
        model=case_b_model,
        neo4j_points=case_b_points
    )

    # Case C
    case_c_csv = "data_sample/neo4j_csv/千福が丘 3丁目.csv"
    case_c_json = "data_sample/neo4j_csv/Case_c.json"
    case_c_area_csv = "data_sample/neo4j_csv/千福が丘 3丁目Area.csv"
    case_c_area_json = "data_sample/neo4j_csv/Case_c_area.json"
    case_c_google = (35.2002115, 138.8778726)
    case_c_model = (35.201230, 138.881257)
    case_c_points = load_points_from_json(case_c_json)
    if not case_c_points:
        case_c_points = load_points_from_csv(case_c_csv)
    case_c_polygon = load_polygon_from_json(case_c_area_json)
    if not case_c_polygon:
        case_c_polygon = load_polygon_from_csv(case_c_area_csv)
    render_case(
        output_path="results/visualization/one_to_zero_case_c.html",
        google=case_c_google,
        model=case_c_model,
        neo4j_points=case_c_points,
        polygon=case_c_polygon
    )

    print("Saved:")
    print("- results/visualization/one_to_zero_case_a.html")
    print("- results/visualization/one_to_zero_case_b.html")
    print("- results/visualization/one_to_zero_case_c.html")


if __name__ == "__main__":
    main()
