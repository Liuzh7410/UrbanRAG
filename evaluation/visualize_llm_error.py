"""
Visualize LLM-only geocoding error characteristics.

Generates:
1) CDF of distance error
2) Accuracy@k bar chart (10/50/100/500/1000m)
3) Error histogram (log-scaled x-axis)
4) Box plot of error distribution

Default input:
  results/baseline/evaluation_results_llm_tokyo_10000.csv
"""

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_errors(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "distance_error_m" not in df.columns:
        raise ValueError(f"Column 'distance_error_m' not found in {csv_path}")

    errors = pd.to_numeric(df["distance_error_m"], errors="coerce").dropna()
    errors = errors[errors >= 0]
    if errors.empty:
        raise ValueError("No valid non-negative error values found")
    return errors.to_numpy(dtype=float)


def plot_cdf(errors: np.ndarray, output_path: str):
    sorted_errors = np.sort(errors)
    y = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100.0

    plt.figure(figsize=(8, 5), dpi=180)
    plt.plot(sorted_errors, y, color="#1f77b4", linewidth=2)
    plt.xlabel("Distance Error (m)")
    plt.ylabel("Cumulative Percentage (%)")
    plt.title("CDF of Distance Error (Accuracy @ K)")
    x_max = max(1000, float(np.max(sorted_errors)))
    plt.xlim(0, x_max)
    plt.ylim(0, 100)

    # Reference lines: 500m and 1000m
    for k in [500, 1000]:
        pct = float(np.mean(errors <= k) * 100)
        plt.axvline(x=k, color="red", linestyle="--", linewidth=1, alpha=0.7)
        plt.axhline(y=pct, color="red", linestyle="--", linewidth=1, alpha=0.4)
        plt.text(k + 8, min(98, pct + 1.5), f"{k}m: {pct:.1f}%", color="red", fontsize=8)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_accuracy_at_k(errors: np.ndarray, output_path: str, thresholds: List[int]):
    total = len(errors)
    counts = [int(np.sum(errors <= k)) for k in thresholds]
    ratios = [c / total * 100 for c in counts]

    labels = [f"@{k}m" for k in thresholds]

    plt.figure(figsize=(8, 5), dpi=180)
    bars = plt.bar(labels, ratios, color="#2ca02c", alpha=0.85)
    for bar, c, r in zip(bars, counts, ratios):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{r:.1f}%\n({c})", ha="center", va="bottom", fontsize=8)
    plt.ylim(0, min(100, max(ratios) + 12))
    plt.ylabel("Percentage (%)")
    plt.title("Accuracy@k of LLM-only Geocoding")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_histogram(errors: np.ndarray, output_path: str):
    max_v = float(np.max(errors))
    bins = np.linspace(0, max_v, 40)
    plt.figure(figsize=(8, 5), dpi=180)
    plt.hist(errors, bins=bins, color="#ff7f0e", alpha=0.85, edgecolor="white")
    plt.xlabel("Distance Error (m)")
    plt.ylabel("Count")
    plt.title("Histogram of LLM-only Geocoding Error")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize LLM-only geocoding errors")
    parser.add_argument("--input", default="results/baseline/evaluation_results_llm_tokyo_10000.csv")
    parser.add_argument("--output-dir", default="visualization")
    args = parser.parse_args()

    errors = load_errors(args.input)

    os.makedirs(args.output_dir, exist_ok=True)

    cdf_path = os.path.join(args.output_dir, "llm_error_cdf.png")
    acc_path = os.path.join(args.output_dir, "llm_error_accuracy_at_k.png")
    hist_path = os.path.join(args.output_dir, "llm_error_hist.png")

    thresholds = [10, 50, 100, 500, 1000]

    plot_cdf(errors, cdf_path)
    plot_accuracy_at_k(errors, acc_path, thresholds)
    plot_histogram(errors, hist_path)

    print("Saved:")
    print(f"- {cdf_path}")
    print(f"- {acc_path}")
    print(f"- {hist_path}")


if __name__ == "__main__":
    main()
