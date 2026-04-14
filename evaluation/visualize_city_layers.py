"""
Visualize ward/area/road/poi for each city on a single static map.

This version outputs compact PNG files (instead of huge interactive HTML)
so they are easier to open and share.
"""

import os
from typing import Dict

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely import wkt


CITY_CONFIG: Dict[str, Dict[str, str]] = {
    "shizuoka": {
        "ward": "/Users/liuziheng/Documents/Shizuoka UrbanKG/Processed data/Administrative/Shizuoka.shp",
        "area": "/Users/liuziheng/Documents/Shizuoka UrbanKG/Processed data/Administrative/Shizuoka_towns.shp",
        "poi": "/Users/liuziheng/Documents/Shizuoka UrbanKG/Processed data/POI/Shizuoka_POI.csv",
        "road": "/Users/liuziheng/Documents/Shizuoka UrbanKG/Processed data/Railway/DRM_Shizuoka.shp",
    },
    "susono": {
        "ward": "/Users/liuziheng/Documents/Susono UrbanKG/Processed data/Administrative/Susuno.shp",
        "area": "/Users/liuziheng/Documents/Susono UrbanKG/Processed data/Administrative/Susono_towns.shp",
        "poi": "/Users/liuziheng/Documents/Susono UrbanKG/Processed data/POI/Susono_POI.csv",
        "road": "/Users/liuziheng/Documents/Susono UrbanKG/Processed data/Railway/DRM_Susono.shp",
    },
    "tokyo": {
        "ward": "/Users/liuziheng/Documents/Tokyo UrbanKG/Processed data/Administrative/Tokyo23.shp",
        "area": "/Users/liuziheng/Documents/Tokyo UrbanKG/Processed data/Administrative/Tokyo23_towns.shp",
        "poi": "/Users/liuziheng/Documents/Tokyo UrbanKG/Processed data/POI/Tokyo_POI.csv",
        "road": "/Users/liuziheng/Documents/Tokyo UrbanKG/Processed data/Railway/DRM_Tokyo.shp",
    },
}


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")


def load_poi(csv_path: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    if "geometry" not in df.columns:
        raise ValueError(f"POI CSV missing geometry column: {csv_path}")
    df["geometry"] = df["geometry"].apply(lambda x: wkt.loads(x) if isinstance(x, str) else None)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf.dropna(subset=["geometry"])


def visualize_city(city: str, cfg: Dict[str, str], output_dir: str, max_poi: int = 30000):
    for key, path in cfg.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {key} file for {city}: {path}")

    ward = ensure_wgs84(gpd.read_file(cfg["ward"]))
    area = ensure_wgs84(gpd.read_file(cfg["area"]))
    road = ensure_wgs84(gpd.read_file(cfg["road"]))
    poi = load_poi(cfg["poi"])

    if len(poi) > max_poi:
        poi = poi.sample(max_poi, random_state=42)

    fig, ax = plt.subplots(figsize=(11, 11), dpi=180)
    fig.patch.set_facecolor("#edf0f2")
    ax.set_facecolor("#edf0f2")

    # All layers in one map, style close to your reference.
    area.boundary.plot(ax=ax, color="lightblue", linewidth=0.8, alpha=0.9, zorder=1)
    road.plot(ax=ax, color="#6d6d6d", linewidth=0.6, alpha=0.7, zorder=2)
    poi.plot(ax=ax, color="#7fa6ff", markersize=3, marker="s", alpha=0.35, zorder=3)
    ward.boundary.plot(ax=ax, color="#2e41ff", linewidth=1.6, alpha=0.95, zorder=4)

    ax.set_axis_off()
    ax.set_aspect("equal")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{city}_layers_map.png")
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    output_dir = "results/visualization"
    for city, cfg in CITY_CONFIG.items():
        visualize_city(city, cfg, output_dir)


if __name__ == "__main__":
    main()
