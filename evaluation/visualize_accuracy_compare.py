"""
Reproduce three accuracy comparison figures for LLM / RAG / LightRAG / GraphRAG / UrbanRAG:
1) Violin distributions (3 panels: Tokyo/Shizuoka/Susono)
2) Sample-size scaling (3x3: city x percentile)
3) Dumbbell P50->P95 (3 panels: Tokyo/Shizuoka/Susono)
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.facecolor"] = "none"
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["savefig.facecolor"] = "none"
plt.rcParams["savefig.transparent"] = True

CITIES = ["tokyo", "shizuoka", "susono"]
METHODS = ["LLM", "RAG", "LightRAG", "GraphRAG", "UrbanRAG"]
METHOD_COLORS = {
    "LLM": "#E74C3C",
    "RAG": "#3498DB",
    "LightRAG": "#9B59B6",
    "GraphRAG": "#F39C12",
    "UrbanRAG": "#27AE60",
}
CITY_DISPLAY = {"tokyo": "Tokyo", "shizuoka": "Shizuoka", "susono": "Susono"}
METHOD_MARKERS = {"P50": "o", "P75": "s", "P95": "D"}

# Manual stats provided by user (server-side consolidated values).
MANUAL_LIST_STATS = {
    "LLM": {
        "Tokyo": {100: {"P50": 313.00, "P75": 500.60, "P95": 850.05}, 1000: {"P50": 279.94, "P75": 410.07, "P95": 690.02}, 10000: {"P50": 243.28, "P75": 365.64, "P95": 654.97}},
        "Shizuoka": {100: {"P50": 1467.39, "P75": 2636.09, "P95": 7948.29}, 1000: {"P50": 1647.61, "P75": 2860.77, "P95": 5703.08}, 10000: {"P50": 1598.37, "P75": 2845.31, "P95": 5788.71}},
        "Susono": {100: {"P50": 2173.81, "P75": 4005.34, "P95": 8083.84}, 1000: {"P50": 1982.79, "P75": 3548.09, "P95": 7391.41}, 10000: {"P50": None, "P75": None, "P95": None}},
    },
    "RAG": {
        "Tokyo": {100: {"P50": 36.80, "P75": 73.38, "P95": 307.81}, 1000: {"P50": 40.40, "P75": 95.52, "P95": 376.66}, 10000: {"P50": 44.50, "P75": 99.00, "P95": 412.45}},
        "Shizuoka": {100: {"P50": 50.20, "P75": 90.56, "P95": 932.54}, 1000: {"P50": 50.77, "P75": 116.82, "P95": 917.00}, 10000: {"P50": 73.55, "P75": 217.89, "P95": 2449.34}},
        "Susono": {100: {"P50": 96.50, "P75": 252.10, "P95": 1214.19}, 1000: {"P50": 122.34, "P75": 291.16, "P95": 1280.16}, 10000: {"P50": None, "P75": None, "P95": None}},
    },
    "LightRAG": {
        "Tokyo": {100: {"P50": 267.38, "P75": 473.95, "P95": 704.39}, 1000: {"P50": 261.73, "P75": 453.48, "P95": 866.12}, 10000: {"P50": 234.13, "P75": 414.65, "P95": 810.80}},
        "Shizuoka": {100: {"P50": 623.95, "P75": 1808.58, "P95": 4427.15}, 1000: {"P50": 462.47, "P75": 1898.77, "P95": 4632.23}, 10000: {"P50": 852.92, "P75": 2925.20, "P95": 7802.67}},
        "Susono": {100: {"P50": 288.01, "P75": 950.57, "P95": 2536.95}, 1000: {"P50": 327.24, "P75": 977.15, "P95": 2623.88}, 10000: {"P50": None, "P75": None, "P95": None}},
    },
    "UrbanRAG": {
        "Tokyo": {100: {"P50": 7.74, "P75": 37.01, "P95": 130.28}, 1000: {"P50": 10.30, "P75": 40.81, "P95": 250.83}, 10000: {"P50": 7.89, "P75": 34.97, "P95": 183.74}},
        "Shizuoka": {100: {"P50": 15.10, "P75": 50.71, "P95": 523.51}, 1000: {"P50": 9.61, "P75": 54.18, "P95": 329.53}, 10000: {"P50": 18.04, "P75": 75.34, "P95": 396.50}},
        "Susono": {100: {"P50": 73.10, "P75": 318.66, "P95": 1166.54}, 1000: {"P50": 50.96, "P75": 276.72, "P95": 1238.83}, 10000: {"P50": None, "P75": None, "P95": None}},
    },
}


def _extract_size_from_filename(path: str) -> int:
    m = re.search(r"_(\d+)\.csv$", os.path.basename(path))
    if m:
        return int(m.group(1))
    return 100


def _load_error_array(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if "distance_error_m" not in df.columns:
        return np.array([], dtype=float)
    arr = pd.to_numeric(df["distance_error_m"], errors="coerce").dropna()
    arr = arr[arr >= 0]
    return arr.to_numpy(dtype=float)


def load_all_results(results_dir: str, baseline_dir: str) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
    data: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {
        city: {method: {} for method in METHODS} for city in CITIES
    }

    urban_files = glob.glob(os.path.join(results_dir, "evaluation_results_*.csv"))
    for path in urban_files:
        b = os.path.basename(path)
        if (
            b.startswith("evaluation_results_llm_")
            or b.startswith("evaluation_results_rag_")
            or b.startswith("evaluation_results_lightrag_")
            or b.startswith("evaluation_results_graphrag_")
        ):
            continue
        m = re.match(r"evaluation_results_(tokyo|shizuoka|susono)(?:_\d+)?\.csv$", b)
        if not m:
            continue
        city = m.group(1)
        n = _extract_size_from_filename(path)
        data[city]["UrbanRAG"][n] = _load_error_array(path)

    base_files = glob.glob(os.path.join(baseline_dir, "evaluation_results_*.csv"))
    for path in base_files:
        b = os.path.basename(path)
        m = re.match(r"evaluation_results_(llm|rag|lightrag|graphrag)_(tokyo|shizuoka|susono)(?:_\d+)?\.csv$", b)
        if not m:
            continue
        method_raw, city = m.group(1), m.group(2)
        method = {"llm": "LLM", "rag": "RAG", "lightrag": "LightRAG", "graphrag": "GraphRAG"}[method_raw]
        n = _extract_size_from_filename(path)
        data[city][method][n] = _load_error_array(path)

    return data


def _choose_for_violin(sample_dict: Dict[int, np.ndarray]) -> Tuple[int, np.ndarray]:
    if not sample_dict:
        return 0, np.array([], dtype=float)
    target = 1000
    if target in sample_dict:
        return target, sample_dict[target]
    n = max(sample_dict.keys())
    return n, sample_dict[n]


def _percentiles(errors: np.ndarray) -> Dict[str, float]:
    if errors.size == 0:
        return {"P50": np.nan, "P75": np.nan, "P95": np.nan}
    return {
        "P50": float(np.percentile(errors, 50)),
        "P75": float(np.percentile(errors, 75)),
        "P95": float(np.percentile(errors, 95)),
    }


def _canonical_method(name: str) -> Optional[str]:
    n = str(name).strip().lower()
    mapping = {
        "llm": "LLM",
        "rag": "RAG",
        "lightrag": "LightRAG",
        "graphrag": "GraphRAG",
        "urbanrag": "UrbanRAG",
    }
    return mapping.get(n)


def load_manual_stats(stats_csv: Optional[str]) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    out: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {
        city: {method: {} for method in METHODS} for city in CITIES
    }
    if not stats_csv:
        return out
    if not os.path.exists(stats_csv):
        raise FileNotFoundError(f"Manual stats CSV not found: {stats_csv}")

    df = pd.read_csv(stats_csv)
    required = {"city", "method", "sample_size", "p50", "p75", "p95"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Manual stats CSV missing columns: {sorted(miss)}")

    for _, row in df.iterrows():
        city = str(row["city"]).strip().lower()
        method = _canonical_method(row["method"])
        if city not in CITIES or method not in METHODS:
            continue
        n = int(row["sample_size"])
        out[city][method][n] = {
            "P50": float(row["p50"]),
            "P75": float(row["p75"]),
            "P95": float(row["p95"]),
        }
    return out


def load_manual_stats_from_list() -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    out: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {
        city: {method: {} for method in METHODS} for city in CITIES
    }
    for method, city_map in MANUAL_LIST_STATS.items():
        if method not in METHODS:
            continue
        for city_name, sample_map in city_map.items():
            city = city_name.strip().lower()
            if city not in CITIES:
                continue
            for n, p in sample_map.items():
                if p.get("P50") is None or p.get("P75") is None or p.get("P95") is None:
                    continue
                out[city][method][int(n)] = {
                    "P50": float(p["P50"]),
                    "P75": float(p["P75"]),
                    "P95": float(p["P95"]),
                }
    return out


def _get_sample_sizes(
    city: str,
    method: str,
    data: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    manual_stats: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
) -> List[int]:
    a = set(data[city][method].keys())
    b = set(manual_stats[city][method].keys())
    return sorted(a.union(b))


def _get_percentiles(
    city: str,
    method: str,
    n: int,
    data: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    manual_stats: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
) -> Dict[str, float]:
    manual = manual_stats.get(city, {}).get(method, {}).get(n)
    if manual is not None:
        return manual
    arr = data.get(city, {}).get(method, {}).get(n, np.array([], dtype=float))
    return _percentiles(arr)


def _synthetic_distribution_from_percentiles(p: Dict[str, float], size: int = 1200) -> np.ndarray:
    p50, p75, p95 = p["P50"], p["P75"], p["P95"]
    if p50 <= 0 or p75 <= 0 or p95 <= 0:
        return np.array([], dtype=float)
    seg1 = np.full(size // 2, p50)
    seg2 = np.linspace(p50, p75, size // 4, endpoint=True)
    seg3 = np.linspace(p75, p95, size // 5, endpoint=True)
    seg4 = np.linspace(p95, p95 * 1.08, size - len(seg1) - len(seg2) - len(seg3), endpoint=True)
    arr = np.concatenate([seg1, seg2, seg3, seg4]).astype(float)
    arr[arr <= 0] = min(v for v in [p50, p75, p95] if v > 0)
    return arr


def plot_violin(
    data: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    manual_stats: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
    out_path: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=160, constrained_layout=True)
    fig.patch.set_alpha(0.0)
    # fig.suptitle("Violin Plot: Error Distributions (Sample Size = 1000)", fontsize=16, fontweight="bold")

    for i, city in enumerate(CITIES):
        ax = axes[i]
        ax.set_facecolor("none")
        violin_data = []
        labels = []
        colors = []
        for method in METHODS:
            all_sizes = _get_sample_sizes(city, method, data, manual_stats)
            if not all_sizes:
                n, arr = 0, np.array([np.nan], dtype=float)
            else:
                n = 1000 if 1000 in all_sizes else max(all_sizes)
                arr = data[city][method].get(n, np.array([], dtype=float))
                if manual_stats[city][method].get(n):
                    arr = _synthetic_distribution_from_percentiles(manual_stats[city][method][n])
                if arr.size == 0:
                    arr = np.array([np.nan], dtype=float)
            violin_data.append(arr)
            labels.append(method)
            colors.append(METHOD_COLORS[method])

        parts = ax.violinplot(violin_data, showmeans=False, showmedians=False, showextrema=True)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.65)
            body.set_edgecolor(color)

        p50 = [np.nanpercentile(v, 50) if np.isfinite(np.nanmean(v)) else np.nan for v in violin_data]
        p75 = [np.nanpercentile(v, 75) if np.isfinite(np.nanmean(v)) else np.nan for v in violin_data]
        for x, m50, m75, color in zip(range(1, len(METHODS) + 1), p50, p75, colors):
            if np.isnan(m50):
                continue
            ax.plot([x - 0.12, x + 0.12], [m50, m50], color="#1f77b4", lw=2)
            ax.plot([x - 0.09, x + 0.09], [m75, m75], color="#1f77b4", lw=1.6)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yscale("log")
        ax.set_ylabel("Error (meters)", fontsize=12, fontname="Times New Roman")
        ax.set_title(CITY_DISPLAY[city], fontsize=15, fontweight="bold")
        ax.set_xlabel("", fontname="Times New Roman")
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, axis="y", alpha=0.22)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname("Times New Roman")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, transparent=True)
    plt.close(fig)


def plot_scaling(
    data: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    manual_stats: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
    out_path: str,
):
    fig, axes = plt.subplots(3, 3, figsize=(16, 14), dpi=160, constrained_layout=True)
    fig.patch.set_alpha(0.0)
    # fig.suptitle("Sample Size Scaling: How Error Changes with Training Data", fontsize=16, fontweight="bold")
    metrics = ["P50", "P75", "P95"]

    for r, city in enumerate(CITIES):
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            ax.set_facecolor("none")
            for method in METHODS:
                samples = _get_sample_sizes(city, method, data, manual_stats)
                xs, ys = [], []
                for n in samples:
                    p = _get_percentiles(city, method, n, data, manual_stats)[metric]
                    if np.isnan(p):
                        continue
                    xs.append(n)
                    ys.append(p)
                if xs:
                    ax.plot(xs, ys, marker="o", linewidth=1.8, color=METHOD_COLORS[method], label=method)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{CITY_DISPLAY[city]} - {metric}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Sample Size", fontname="Times New Roman")
            ax.set_ylabel("Error (meters)", fontname="Times New Roman")
            ax.grid(True, alpha=0.22)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontname("Times New Roman")
            if r == 0 and c == 2:
                ax.legend(loc="upper right", fontsize=10)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, transparent=True)
    plt.close(fig)


def plot_dumbbell(
    data: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    manual_stats: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
    out_path: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), dpi=160, constrained_layout=True)
    fig.patch.set_alpha(0.0)
    # fig.suptitle("Dumbbell Plot: Error Distribution (P50 -> P95)", fontsize=16, fontweight="bold")
    global_x_values: List[float] = []

    for i, city in enumerate(CITIES):
        ax = axes[i]
        ax.set_facecolor("none")
        y_labels = []
        y_values = []
        y = 0
        for method in METHODS:
            samples = sorted(_get_sample_sizes(city, method, data, manual_stats), reverse=True)
            for n in samples:
                p = _get_percentiles(city, method, n, data, manual_stats)
                if np.isnan(p["P50"]):
                    continue
                y_labels.append(f"{method} (n={n})")
                y_values.append((y, method, p))
                global_x_values.extend([p["P50"], p["P75"], p["P95"]])
                y += 1

        for y_idx, method, p in y_values:
            col = METHOD_COLORS[method]
            ax.plot([p["P50"], p["P95"]], [y_idx, y_idx], color=col, alpha=0.55, linewidth=2)
            ax.scatter(p["P50"], y_idx, color=col, marker=METHOD_MARKERS["P50"], s=95, alpha=0.9)
            ax.scatter(p["P75"], y_idx, color=col, marker=METHOD_MARKERS["P75"], s=80, alpha=0.6)
            ax.scatter(p["P95"], y_idx, color=col, marker=METHOD_MARKERS["P95"], s=130, alpha=0.45)

        ax.set_xscale("log")
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Error (meters)", fontsize=12, fontname="Times New Roman")
        ax.set_title(CITY_DISPLAY[city], fontsize=15, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.2)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname("Times New Roman")

    # Force identical x-axis range/ticks for all three city subplots.
    if global_x_values:
        positive = [v for v in global_x_values if v > 0]
        if positive:
            min_pow = int(np.floor(np.log10(min(positive))))
            max_pow = int(np.ceil(np.log10(max(positive))))
            ticks = [10 ** p for p in range(min_pow, max_pow + 1)]
            x_min = (10 ** min_pow) * 0.8
            x_max = (10 ** max_pow) * 1.2
            for ax in axes:
                ax.set_xlim(x_min, x_max)
                ax.set_xticks(ticks)

    legend_handles = [
        plt.Line2D([0], [0], marker=METHOD_MARKERS["P50"], color="w", markerfacecolor="gray", markersize=10, label="P50"),
        plt.Line2D([0], [0], marker=METHOD_MARKERS["P75"], color="w", markerfacecolor="gray", markersize=9, label="P75"),
        plt.Line2D([0], [0], marker=METHOD_MARKERS["P95"], color="w", markerfacecolor="gray", markersize=10, label="P95"),
    ]
    # Put percentile legend inside the right-most subplot (bottom-right corner).
    axes[-1].legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        borderaxespad=0.4,
        frameon=True,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, transparent=True)
    plt.close(fig)


def print_summary(
    data: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    manual_stats: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
):
    print("\nSummary percentiles:")
    for city in CITIES:
        print(f"\n[{CITY_DISPLAY[city]}]")
        for method in METHODS:
            samples = _get_sample_sizes(city, method, data, manual_stats)
            if not samples:
                print(f"  {method}: no data")
                continue
            for n in samples:
                p = _get_percentiles(city, method, n, data, manual_stats)
                print(
                    f"  {method} n={n}: "
                    f"P50={p['P50']:.2f}, P75={p['P75']:.2f}, P95={p['P95']:.2f}"
                )


def main():
    parser = argparse.ArgumentParser(description="Reproduce accuracy comparison figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--baseline-dir", default="results/baseline")
    parser.add_argument("--out-dir", default="visualization")
    parser.add_argument("--stats-csv", default="", help="Optional manual stats CSV: city,method,sample_size,p50,p75,p95")
    parser.add_argument("--use-manual-list", action="store_true", help="Force using built-in manual list stats")
    args = parser.parse_args()

    data = load_all_results(args.results_dir, args.baseline_dir)
    # Priority: explicit --use-manual-list > --stats-csv > built-in manual list.
    if args.use_manual_list:
        manual_stats = load_manual_stats_from_list()
    elif args.stats_csv:
        manual_stats = load_manual_stats(args.stats_csv)
    else:
        manual_stats = load_manual_stats_from_list()
    os.makedirs(args.out_dir, exist_ok=True)

    out_violin = os.path.join(args.out_dir, "accuracy_compare_violin.png")
    out_scaling = os.path.join(args.out_dir, "accuracy_compare_scaling.png")
    out_dumbbell = os.path.join(args.out_dir, "accuracy_compare_dumbbell.png")

    plot_violin(data, manual_stats, out_violin)
    plot_scaling(data, manual_stats, out_scaling)
    plot_dumbbell(data, manual_stats, out_dumbbell)
    print_summary(data, manual_stats)

    print("\nSaved figures:")
    print(f"  {out_violin}")
    print(f"  {out_scaling}")
    print(f"  {out_dumbbell}")


if __name__ == "__main__":
    main()
