"""
Visualization script for evaluation results
Generate plots and summary statistics
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import statistics
import os

# Set font for Japanese text support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_evaluation_results(csv_file):
    """Load evaluation results from CSV"""
    results = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['distance_error_m'] = float(row['distance_error_m'])
            row['our_latitude'] = float(row['our_latitude'])
            row['our_longitude'] = float(row['our_longitude'])
            row['google_latitude'] = float(row['google_latitude'])
            row['google_longitude'] = float(row['google_longitude'])
            results.append(row)
    return results


def plot_error_distribution(results, output_file):
    """Plot error distribution histogram"""
    errors = [r['distance_error_m'] for r in results]

    plt.figure(figsize=(10, 6))

    # Create histogram
    plt.hist(errors, bins=20, color='steelblue', alpha=0.7, edgecolor='black')

    # Add mean and median lines
    mean_error = statistics.mean(errors)
    median_error = statistics.median(errors)

    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_error:.2f}m')
    plt.axvline(median_error, color='green', linestyle='--', linewidth=2,
                label=f'Median: {median_error:.2f}m')

    plt.xlabel('Distance Error (meters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Geocoding Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Error distribution plot saved to: {output_file}")
    plt.close()


def plot_error_by_category(results, output_file):
    """Plot error ranges distribution"""
    error_ranges = [
        (0, 10, "< 10m\n(Excellent)"),
        (10, 25, "10-25m\n(Very Good)"),
        (25, 50, "25-50m\n(Good)"),
        (50, 100, "50-100m\n(Acceptable)"),
        (100, 200, "100-200m\n(Fair)"),
        (200, float('inf'), "> 200m\n(Poor)")
    ]

    errors = [r['distance_error_m'] for r in results]
    counts = []
    labels = []
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#95a5a6']

    for min_err, max_err, label in error_ranges:
        count = sum(1 for e in errors if min_err <= e < max_err)
        counts.append(count)
        labels.append(label)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(counts)), counts, color=colors, alpha=0.7, edgecolor='black')

    plt.xlabel('Error Range', fontsize=12)
    plt.ylabel('Number of Addresses', fontsize=12)
    plt.title('Error Distribution by Category', fontsize=14, fontweight='bold')
    plt.xticks(range(len(labels)), labels, fontsize=9)
    plt.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/len(errors)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Error category plot saved to: {output_file}")
    plt.close()


def plot_spatial_comparison(results, output_file):
    """Plot our results vs Google results on a map"""
    fig, ax = plt.subplots(figsize=(12, 10))

    for idx, result in enumerate(results, 1):
        our_lon = result['our_longitude']
        our_lat = result['our_latitude']
        google_lon = result['google_longitude']
        google_lat = result['google_latitude']
        error = result['distance_error_m']

        # Plot our point
        ax.scatter(our_lon, our_lat, c='blue', s=100, marker='o',
                  alpha=0.7, edgecolors='black', linewidth=1.5,
                  label='Our Result' if idx == 1 else '')

        # Plot Google point
        ax.scatter(google_lon, google_lat, c='red', s=100, marker='s',
                  alpha=0.7, edgecolors='black', linewidth=1.5,
                  label='Google Maps' if idx == 1 else '')

        # Draw line connecting them
        ax.plot([our_lon, google_lon], [our_lat, google_lat],
               'k--', alpha=0.5, linewidth=1)

        # Add address label
        ax.annotate(f'#{idx}\n{error:.1f}m',
                   xy=(our_lon, our_lat),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='yellow', alpha=0.5))

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Spatial Comparison: Our Results vs Google Maps',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Spatial comparison plot saved to: {output_file}")
    plt.close()


def plot_error_by_confidence(results, output_file):
    """Plot error statistics by confidence level"""
    confidence_errors = defaultdict(list)

    for result in results:
        confidence_errors[result['confidence']].append(result['distance_error_m'])

    if len(confidence_errors) <= 1:
        print("  ⚠️ Skipping confidence plot (only one confidence level)")
        return

    confidences = sorted(confidence_errors.keys())
    means = [statistics.mean(confidence_errors[c]) for c in confidences]
    medians = [statistics.median(confidence_errors[c]) for c in confidences]

    x = range(len(confidences))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar([i - width/2 for i in x], means, width,
                   label='Mean Error', color='steelblue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar([i + width/2 for i in x], medians, width,
                   label='Median Error', color='lightcoral', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Confidence Level', fontsize=12)
    ax.set_ylabel('Error (meters)', fontsize=12)
    ax.set_title('Error by Confidence Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in confidences])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}m',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Confidence comparison plot saved to: {output_file}")
    plt.close()


def generate_summary_report(results, output_file):
    """Generate text summary report"""
    errors = [r['distance_error_m'] for r in results]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("GEOCODING EVALUATION SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total Addresses Evaluated: {len(results)}\n")
        f.write(f"Evaluation Date: 2025-12-23\n\n")

        f.write("-"*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Error:        {statistics.mean(errors):>10.2f} m\n")
        f.write(f"Median Error:      {statistics.median(errors):>10.2f} m\n")
        f.write(f"Min Error:         {min(errors):>10.2f} m\n")
        f.write(f"Max Error:         {max(errors):>10.2f} m\n")
        if len(errors) > 1:
            f.write(f"Std Deviation:     {statistics.stdev(errors):>10.2f} m\n")
        f.write("\n")

        # Error distribution
        f.write("-"*80 + "\n")
        f.write("ERROR DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        error_ranges = [
            (0, 10, "< 10m (Excellent)"),
            (10, 25, "10-25m (Very Good)"),
            (25, 50, "25-50m (Good)"),
            (50, 100, "50-100m (Acceptable)"),
            (100, 200, "100-200m (Fair)"),
            (200, float('inf'), "> 200m (Poor)")
        ]

        for min_err, max_err, label in error_ranges:
            count = sum(1 for e in errors if min_err <= e < max_err)
            pct = (count / len(errors)) * 100
            f.write(f"{label:25s}: {count:3d} ({pct:5.1f}%)\n")
        f.write("\n")

        # Detailed results
        f.write("-"*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-"*80 + "\n")

        for idx, result in enumerate(results, 1):
            f.write(f"\n[{idx}] {result['address']}\n")
            f.write(f"    Our:    ({result['our_latitude']:.6f}, {result['our_longitude']:.6f})\n")
            f.write(f"    Google: ({result['google_latitude']:.6f}, {result['google_longitude']:.6f})\n")
            f.write(f"    Error:  {result['distance_error_m']:.2f} m\n")
            f.write(f"    Google Type: {result['google_location_type']}\n")
            f.write(f"    Method: {result['method']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"✓ Summary report saved to: {output_file}")


if __name__ == "__main__":
    input_file = 'results/evaluation_results.csv'
    output_dir = 'results/plots'

    # Check input file
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Please run evaluation.py first.")
        exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    print("\n" + "="*80)
    print("Generating Evaluation Visualizations")
    print("="*80 + "\n")

    results = load_evaluation_results(input_file)
    print(f"📊 Loaded {len(results)} evaluation results\n")

    # Generate plots
    print("Creating visualizations...\n")

    plot_error_distribution(results, f'{output_dir}/error_distribution.png')
    plot_error_by_category(results, f'{output_dir}/error_by_category.png')
    plot_spatial_comparison(results, f'{output_dir}/spatial_comparison.png')
    plot_error_by_confidence(results, f'{output_dir}/error_by_confidence.png')

    # Generate summary report
    print("\nGenerating summary report...\n")
    generate_summary_report(results, 'results/evaluation_summary.txt')

    print("\n" + "="*80)
    print("Visualization Complete!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}/")
