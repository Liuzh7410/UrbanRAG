"""
Visualize Google Maps geocode vs Neo4j POI points for a given address.

Usage:
  python visualize_points.py --address "東京都 文京区 後楽 2丁目 2-8"
"""

import argparse
import os
import re
from typing import List, Tuple

from dotenv import load_dotenv
import folium
import googlemaps
from neo4j import GraphDatabase


def parse_point_wkt(wkt_str: str):
    if not wkt_str:
        return None
    m = re.match(r"^POINT\s*\(([-\d\.]+)\s+([-\d\.]+)\)$", str(wkt_str).strip())
    if not m:
        return None
    lon = float(m.group(1))
    lat = float(m.group(2))
    return lat, lon


def get_google_coords(address: str, gmaps_client) -> Tuple[float, float]:
    result = gmaps_client.geocode(address, language='ja')
    if not result:
        raise RuntimeError(f"Google Maps API: No result for {address}")
    location = result[0]['geometry']['location']
    return location['lat'], location['lng']


def get_poi_coords(address: str, driver) -> List[Tuple[float, float]]:
    coords = []
    cypher = """
    MATCH (p:POI)
    WHERE p.address STARTS WITH $address
    RETURN p.geometry AS geometry
    LIMIT 200
    """
    with driver.session(database=os.getenv('NEO4J_DATABASE')) as session:
        result = session.run(cypher, address=address)
        for record in result:
            geom = record.get('geometry')
            latlon = parse_point_wkt(geom)
            if latlon:
                coords.append(latlon)
    return coords


def main():
    parser = argparse.ArgumentParser(description="Visualize Google vs Neo4j POI points")
    parser.add_argument("--address", required=True, help="Target address")
    parser.add_argument("--output", default="results/visualization/points_map.html")
    args = parser.parse_args()

    load_dotenv()

    gmaps_key = os.getenv('GOOGLE_GEO_API_KEY')
    if not gmaps_key:
        raise RuntimeError("GOOGLE_GEO_API_KEY not set in .env")

    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_user = os.getenv('NEO4J_USERNAME')
    neo4j_pass = os.getenv('NEO4J_PASSWORD')
    if not neo4j_uri or not neo4j_user or not neo4j_pass:
        raise RuntimeError("NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD not set in .env")

    gmaps_client = googlemaps.Client(key=gmaps_key)
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

    try:
        g_lat, g_lon = get_google_coords(args.address, gmaps_client)
        poi_coords = get_poi_coords(args.address, driver)
    finally:
        driver.close()

    fmap = folium.Map(location=[g_lat, g_lon], zoom_start=20, tiles="CartoDB positron")

    folium.CircleMarker(
        location=[g_lat, g_lon],
        radius=6,
        color='red',
        fill=True,
        fill_color='red',
        popup='Google Maps'
    ).add_to(fmap)

    for lat, lon in poi_coords:
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color='black',
            fill=True,
            fill_color='black',
            popup='Neo4j POI'
        ).add_to(fmap)

    folium.CircleMarker(
        location=[35.703787, 139.745106],
        radius=6,
        color='blue',
        fill=True,
        fill_color='blue',
        popup='Reference'
    ).add_to(fmap)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fmap.save(args.output)

    print(f"Google Maps: ({g_lat:.6f}, {g_lon:.6f})")
    print(f"Neo4j POI points: {len(poi_coords)}")
    print(f"Map saved to: {args.output}")


if __name__ == "__main__":
    main()
