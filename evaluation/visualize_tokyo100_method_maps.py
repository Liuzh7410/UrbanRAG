"""
Generate 6 maps for Tokyo 100-sample experiments:
1) Ground truth (all green points)
2) LLM-only (error-colored points: green->red)
3) RAG (error-colored points: green->red)
4) LightRAG (error-colored points: green->red)
5) GraphRAG (error-colored points: green->red)
6) UrbanRAG (error-colored points: green->red)
"""

import argparse
import os
from typing import Dict, List

import folium
import pandas as pd


def load_eval_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    required = {
        "address",
        "our_latitude",
        "our_longitude",
        "google_latitude",
        "google_longitude",
        "distance_error_m",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    for col in ["our_latitude", "our_longitude", "google_latitude", "google_longitude", "distance_error_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["address", "google_latitude", "google_longitude"])
    return df


def get_center(df: pd.DataFrame) -> List[float]:
    return [float(df["google_latitude"].mean()), float(df["google_longitude"].mean())]


def draw_ground_truth_map(gt_df: pd.DataFrame, center: List[float], out_html: str):
    fmap = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    for _, row in gt_df.iterrows():
        folium.CircleMarker(
            location=[row["google_latitude"], row["google_longitude"]],
            radius=4,
            color="#2ecc71",
            fill=True,
            fill_color="#2ecc71",
            fill_opacity=0.75,
            weight=0.6,
            popup=f"GT: {row['address']}",
        ).add_to(fmap)
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fmap.save(out_html)


def draw_method_map(
    method_name: str,
    df: pd.DataFrame,
    center: List[float],
    out_html: str,
):
    fmap = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    df = df.dropna(subset=["our_latitude", "our_longitude", "distance_error_m"])
    for _, row in df.iterrows():
        err = float(row["distance_error_m"])
        color, band = error_to_color_band(err)
        popup = (
            f"{method_name}<br>"
            f"Address: {row['address']}<br>"
            f"Error: {err:.2f} m<br>"
            f"Band: {band}"
        )
        folium.CircleMarker(
            location=[row["our_latitude"], row["our_longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=0.7,
            popup=popup,
        ).add_to(fmap)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fmap.save(out_html)


def error_to_color_band(err_m: float):
    # Discrete buckets to match evaluation criteria.
    if err_m < 10:
        return "#00a651", "< 10m (Excellent)"
    if err_m < 25:
        return "#7ac943", "10-25m (Very Good)"
    if err_m < 50:
        return "#cddc39", "25-50m (Good)"
    if err_m < 100:
        return "#ffca28", "50-100m (Acceptable)"
    if err_m < 200:
        return "#ff8f00", "100-200m (Fair)"
    return "#d32f2f", "> 200m (Poor)"


def main():
    parser = argparse.ArgumentParser(description="Visualize Tokyo 100-sample method maps")
    parser.add_argument("--llm", default="results/baseline/evaluation_results_llm_tokyo.csv")
    parser.add_argument("--rag", default="results/baseline/evaluation_results_rag_tokyo.csv")
    parser.add_argument("--lightrag", default="results/baseline/evaluation_results_lightrag_tokyo.csv")
    parser.add_argument("--graphrag", default="results/baseline/evaluation_results_graphrag_tokyo.csv")
    parser.add_argument("--urbanrag", default="results/evaluation_results_tokyo.csv")
    parser.add_argument("--out-dir", default="results/visualization")
    args = parser.parse_args()

    paths: Dict[str, str] = {
        "LLM-only": args.llm,
        "RAG": args.rag,
        "LightRAG": args.lightrag,
        "GraphRAG": args.graphrag,
        "UrbanRAG": args.urbanrag,
    }
    dfs = {k: load_eval_csv(v) for k, v in paths.items()}

    # Ground truth can come from any one evaluation file (Google coordinates are same schema).
    gt_df = next(iter(dfs.values()))
    center = get_center(gt_df)

    out_gt = os.path.join(args.out_dir, "tokyo100_ground_truth.html")
    draw_ground_truth_map(gt_df, center, out_gt)

    outputs = {
        "LLM-only": os.path.join(args.out_dir, "tokyo100_llm_only.html"),
        "RAG": os.path.join(args.out_dir, "tokyo100_rag.html"),
        "LightRAG": os.path.join(args.out_dir, "tokyo100_lightrag.html"),
        "GraphRAG": os.path.join(args.out_dir, "tokyo100_graphrag.html"),
        "UrbanRAG": os.path.join(args.out_dir, "tokyo100_urbanrag.html"),
    }
    for method, out_html in outputs.items():
        draw_method_map(method, dfs[method], center, out_html)

    print("Saved maps:")
    print(f"  {out_gt}")
    for out_html in outputs.values():
        print(f"  {out_html}")


if __name__ == "__main__":
    main()
