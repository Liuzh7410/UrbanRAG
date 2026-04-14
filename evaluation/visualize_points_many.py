"""
Visualize Google Maps geocode vs jittered points from CSV.

Usage:
  python test/visualize_points_copy.py --address "東京都 文京区 後楽 2丁目 2-8"
"""

import argparse
import csv
import os
from typing import List, Tuple

from dotenv import load_dotenv
import folium
import googlemaps


def get_google_coords(address: str, gmaps_client) -> Tuple[float, float]:
    result = gmaps_client.geocode(address, language='ja')
    if not result:
        raise RuntimeError(f"Google Maps API: No result for {address}")
    location = result[0]['geometry']['location']
    return location['lat'], location['lng']


def load_jitter_points(csv_path: str) -> List[Tuple[float, float]]:
    points = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lon = float(row.get("jitter_lon"))
                lat = float(row.get("jitter_lat"))
                points.append((lat, lon))
            except (TypeError, ValueError):
                continue
    return points


def main():
    parser = argparse.ArgumentParser(description="Visualize Google vs jittered CSV points")
    parser.add_argument("--address", required=True, help="Target address")
    parser.add_argument("--csv", default="data_sample/neo4j_csv/neo4j_query_table_data_2026-1-29_jittered.csv")
    parser.add_argument("--output", default="results/visualization/points_map_many.html")
    args = parser.parse_args()

    load_dotenv()
    gmaps_key = os.getenv('GOOGLE_GEO_API_KEY')
    if not gmaps_key:
        raise RuntimeError("GOOGLE_GEO_API_KEY not set in .env")

    gmaps_client = googlemaps.Client(key=gmaps_key)

    g_lat, g_lon = get_google_coords(args.address, gmaps_client)
    jitter_points = load_jitter_points(args.csv)

    fmap = folium.Map(location=[g_lat, g_lon], zoom_start=20, tiles="CartoDB positron")

    # Google Maps point (red)
    folium.CircleMarker(
        location=[g_lat, g_lon],
        radius=6,
        color='red',
        fill=True,
        fill_color='red',
        popup='Google Maps'
    ).add_to(fmap)

    # Jitter points (blue)
    for lat, lon in jitter_points:
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color='black',
            fill=True,
            fill_color='black',
            popup='Jittered'
        ).add_to(fmap)

    # Extra reference point (purple)
    folium.CircleMarker(
        location=[35.687418, 139.729227],
        radius=6,
        color='blue',
        fill=True,
        fill_color='blue',
        popup='Reference'
    ).add_to(fmap)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fmap.save(args.output)

    print(f"Google Maps: ({g_lat:.6f}, {g_lon:.6f})")
    print(f"Jitter points: {len(jitter_points)}")
    print(f"Map saved to: {args.output}")


if __name__ == "__main__":
    main()
