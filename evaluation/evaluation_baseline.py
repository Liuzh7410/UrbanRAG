"""
Evaluate baseline geocoding outputs against Google Maps API.

- Input: results/baseline/geocoding_*.csv
- Output: same name with prefix geocoding -> evaluation
- Console: Evaluation Statistics + Accuracy@k
"""

import os
import csv
import time
import glob
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import googlemaps
from math import radians, cos, sin, asin, sqrt
import statistics

# Load environment variables
load_dotenv()

# Initialize Google Maps client
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_GEO_API_KEY'))


@dataclass
class EvaluationResult:
    address: str
    our_lat: float
    our_lon: float
    google_lat: float
    google_lon: float
    distance_error_m: float
    google_formatted_address: str
    google_location_type: str


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371000
    return c * r


def geocode_with_google(address: str) -> Optional[Dict]:
    try:
        result = gmaps.geocode(address, language='ja')
        if not result:
            print(f"  ⚠️ Google Maps API: No result for {address}")
            return None
        location = result[0]['geometry']['location']
        formatted_address = result[0]['formatted_address']
        location_type = result[0]['geometry'].get('location_type', 'UNKNOWN')
        return {
            'lat': location['lat'],
            'lon': location['lng'],
            'formatted_address': formatted_address,
            'location_type': location_type
        }
    except Exception as e:
        print(f"  ❌ Google Maps API error for {address}: {str(e)}")
        return None


def display_statistics(results: List[EvaluationResult]):
    if not results:
        print("No results to evaluate!")
        return

    errors = [r.distance_error_m for r in results]

    print("\n【Overall Performance】")
    print(f"  Total addresses evaluated: {len(results)}")
    print(f"  Mean error: {statistics.mean(errors):.2f} m")
    print(f"  Median error: {statistics.median(errors):.2f} m")
    print(f"  Min error: {min(errors):.2f} m")
    print(f"  Max error: {max(errors):.2f} m")
    if len(errors) > 1:
        print(f"  Std deviation: {statistics.stdev(errors):.2f} m")

    print("\n【Error Distribution】")
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
        percentage = (count / len(errors)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    print("\n【Error Percentiles】")
    sorted_errors = sorted(errors)
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        idx = int(len(sorted_errors) * p / 100)
        if idx >= len(sorted_errors):
            idx = len(sorted_errors) - 1
        print(f"  P{p}: {sorted_errors[idx]:.2f} m")

    print("\n【Accuracy @k meters】")
    thresholds = [10, 50, 100, 500, 1000]
    total = len(errors)
    for k in thresholds:
        count = sum(1 for e in errors if e <= k)
        percentage = (count / total) * 100 if total else 0.0
        print(f"  <= {k}m: {count} ({percentage:.1f}%)")


def evaluate_baseline_file(input_csv: str, output_csv: str, rate_limit_delay: float = 0.1):
    print("=" * 80)
    print("Baseline Evaluation - Comparing with Google Maps API")
    print("=" * 80)

    our_results = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            our_results.append(row)

    print(f"\n📊 Loaded {len(our_results)} geocoding results")

    evaluation_results = []

    for idx, result in enumerate(our_results, 1):
        address = result['address']
        our_lat = float(result['latitude'])
        our_lon = float(result['longitude'])

        print(f"\n[{idx}/{len(our_results)}] Evaluating: {address}")
        print(f"  Our coordinates: ({our_lat:.6f}, {our_lon:.6f})")

        google_result = geocode_with_google(address)
        if not google_result:
            print(f"  ⚠️ Skipping (Google Maps failed)")
            continue

        google_lat = google_result['lat']
        google_lon = google_result['lon']
        print(f"  Google coordinates: ({google_lat:.6f}, {google_lon:.6f})")

        distance_error = haversine_distance(our_lat, our_lon, google_lat, google_lon)
        print(f"  Distance error: {distance_error:.2f} meters")

        evaluation_results.append(EvaluationResult(
            address=address,
            our_lat=our_lat,
            our_lon=our_lon,
            google_lat=google_lat,
            google_lon=google_lon,
            distance_error_m=distance_error,
            google_formatted_address=google_result['formatted_address'],
            google_location_type=google_result['location_type']
        ))

        time.sleep(rate_limit_delay)

    print(f"\n{'=' * 80}")
    print("Saving evaluation results...")
    print(f"{'=' * 80}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'address',
            'our_latitude',
            'our_longitude',
            'google_latitude',
            'google_longitude',
            'distance_error_m',
            'google_formatted_address',
            'google_location_type'
        ])
        for result in evaluation_results:
            writer.writerow([
                result.address,
                result.our_lat,
                result.our_lon,
                result.google_lat,
                result.google_lon,
                result.distance_error_m,
                result.google_formatted_address,
                result.google_location_type
            ])

    print(f"✓ Detailed results saved to: {output_csv}")

    print(f"\n{'=' * 80}")
    print("Evaluation Statistics")
    print(f"{'=' * 80}")
    display_statistics(evaluation_results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline geocoding outputs against Google Maps API")
    parser.add_argument("--input", default=None, help="Single geocoding result CSV to evaluate")
    parser.add_argument("--pattern", default=None, help="Glob pattern for multiple geocoding result CSVs")
    parser.add_argument("--rate-limit-delay", type=float, default=0.1)
    parser.add_argument("--only-missing", action="store_true", help="Skip files whose evaluation CSV already exists")
    args = parser.parse_args()
    input_dir = os.path.join('results', 'baseline')

    if args.input:
        files = [args.input]
    else:
        pattern = args.pattern or os.path.join(input_dir, 'geocoding_results_*.csv')
        files = sorted(glob.glob(pattern))

    if not files:
        print(f"No baseline files found.")
        return

    for input_csv in files:
        base = os.path.basename(input_csv)
        output_name = base.replace('geocoding', 'evaluation', 1)
        output_csv = os.path.join(input_dir, output_name)
        if args.only_missing and os.path.exists(output_csv):
            print(f"Skip existing: {output_csv}")
            continue
        evaluate_baseline_file(input_csv, output_csv, rate_limit_delay=args.rate_limit_delay)


if __name__ == '__main__':
    main()
