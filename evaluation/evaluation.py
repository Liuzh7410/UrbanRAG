"""
Geocoding Evaluation Script
Compare our KG-Enhanced Geocoding results with Google Maps API ground truth
Calculate distance errors and generate evaluation metrics
"""

import os
import csv
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
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
    """Evaluation result for a single address"""
    address: str
    our_lat: float
    our_lon: float
    google_lat: float
    google_lon: float
    distance_error_m: float
    confidence: str
    scenario: str
    method: str
    google_formatted_address: str
    google_location_type: str


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Returns:
        Distance in meters
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Radius of earth in meters
    r = 6371000

    return c * r


def geocode_with_google(address: str) -> Optional[Dict]:
    """
    Geocode an address using Google Maps API

    Returns:
        Dict with lat, lon, formatted_address, location_type
        None if geocoding failed
    """
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


def evaluate_geocoding_results(input_csv: str, output_csv: str,
                               rate_limit_delay: float = 0.1):
    """
    Evaluate geocoding results against Google Maps API

    Args:
        input_csv: Path to geocoding results CSV
        output_csv: Path to save evaluation results
        rate_limit_delay: Delay between API calls (seconds)
    """
    print("="*80)
    print("Geocoding Evaluation - Comparing with Google Maps API")
    print("="*80)

    # Read our geocoding results
    our_results = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            our_results.append(row)

    print(f"\n📊 Loaded {len(our_results)} geocoding results")

    # Evaluate each address
    evaluation_results = []

    for idx, result in enumerate(our_results, 1):
        address = result['address']
        if result.get('confidence') == 'failed':
            print(f"\n[{idx}/{len(our_results)}] Skipping failed: {address}")
            continue
        our_lat = float(result['latitude'])
        our_lon = float(result['longitude'])

        print(f"\n[{idx}/{len(our_results)}] Evaluating: {address}")
        print(f"  Our coordinates: ({our_lat:.6f}, {our_lon:.6f})")

        # Get Google Maps result
        google_result = geocode_with_google(address)

        if not google_result:
            print(f"  ⚠️ Skipping (Google Maps failed)")
            continue

        google_lat = google_result['lat']
        google_lon = google_result['lon']

        print(f"  Google coordinates: ({google_lat:.6f}, {google_lon:.6f})")

        # Calculate distance error
        distance_error = haversine_distance(our_lat, our_lon, google_lat, google_lon)

        print(f"  Distance error: {distance_error:.2f} meters")

        # Store evaluation result
        eval_result = EvaluationResult(
            address=address,
            our_lat=our_lat,
            our_lon=our_lon,
            google_lat=google_lat,
            google_lon=google_lon,
            distance_error_m=distance_error,
            confidence=result['confidence'],
            scenario=result['scenario'],
            method=result['method'],
            google_formatted_address=google_result['formatted_address'],
            google_location_type=google_result['location_type']
        )

        evaluation_results.append(eval_result)

        # Rate limiting
        time.sleep(rate_limit_delay)

    # Save detailed evaluation results
    print(f"\n{'='*80}")
    print("Saving evaluation results...")
    print(f"{'='*80}")

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'address',
            'our_latitude',
            'our_longitude',
            'google_latitude',
            'google_longitude',
            'distance_error_m',
            'confidence',
            'scenario',
            'method',
            'google_formatted_address',
            'google_location_type'
        ])

        # Data
        for result in evaluation_results:
            writer.writerow([
                result.address,
                result.our_lat,
                result.our_lon,
                result.google_lat,
                result.google_lon,
                result.distance_error_m,
                result.confidence,
                result.scenario,
                result.method,
                result.google_formatted_address,
                result.google_location_type
            ])

    print(f"✓ Detailed results saved to: {output_csv}")

    # Calculate and display statistics
    print(f"\n{'='*80}")
    print("Evaluation Statistics")
    print(f"{'='*80}")

    display_statistics(evaluation_results)


def display_statistics(results: List[EvaluationResult]):
    """
    Calculate and display evaluation statistics
    """
    if not results:
        print("No results to evaluate!")
        return

    errors = [r.distance_error_m for r in results]

    # Overall statistics
    print("\n【Overall Performance】")
    print(f"  Total addresses evaluated: {len(results)}")
    print(f"  Mean error: {statistics.mean(errors):.2f} m")
    print(f"  Median error: {statistics.median(errors):.2f} m")
    print(f"  Min error: {min(errors):.2f} m")
    print(f"  Max error: {max(errors):.2f} m")
    if len(errors) > 1:
        print(f"  Std deviation: {statistics.stdev(errors):.2f} m")

    # Error distribution
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

    # Statistics by confidence level
    print("\n【Performance by Confidence Level】")
    confidence_groups = {}
    for result in results:
        conf = result.confidence
        if conf not in confidence_groups:
            confidence_groups[conf] = []
        confidence_groups[conf].append(result.distance_error_m)

    for conf in sorted(confidence_groups.keys()):
        errors_group = confidence_groups[conf]
        print(f"\n  {conf.upper()}:")
        print(f"    Count: {len(errors_group)}")
        print(f"    Mean error: {statistics.mean(errors_group):.2f} m")
        print(f"    Median error: {statistics.median(errors_group):.2f} m")

    # Statistics by scenario
    print("\n【Performance by Scenario】")
    scenario_groups = {}
    for result in results:
        scenario = result.scenario
        if scenario not in scenario_groups:
            scenario_groups[scenario] = []
        scenario_groups[scenario].append(result.distance_error_m)

    for scenario in sorted(scenario_groups.keys()):
        errors_group = scenario_groups[scenario]
        print(f"\n  {scenario}:")
        print(f"    Count: {len(errors_group)}")
        print(f"    Mean error: {statistics.mean(errors_group):.2f} m")
        print(f"    Median error: {statistics.median(errors_group):.2f} m")

    # Statistics by Google location type
    print("\n【Performance by Google Location Type】")
    location_type_groups = {}
    for result in results:
        loc_type = result.google_location_type
        if loc_type not in location_type_groups:
            location_type_groups[loc_type] = []
        location_type_groups[loc_type].append(result.distance_error_m)

    for loc_type in sorted(location_type_groups.keys()):
        errors_group = location_type_groups[loc_type]
        print(f"\n  {loc_type}:")
        print(f"    Count: {len(errors_group)}")
        print(f"    Mean error: {statistics.mean(errors_group):.2f} m")
        print(f"    Median error: {statistics.median(errors_group):.2f} m")

    # Percentiles
    print("\n【Error Percentiles】")
    sorted_errors = sorted(errors)
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        idx = int(len(sorted_errors) * p / 100)
        if idx >= len(sorted_errors):
            idx = len(sorted_errors) - 1
        print(f"  P{p}: {sorted_errors[idx]:.2f} m")


if __name__ == "__main__":
    # Configuration
    input_file = 'results/geocoding_results_tokyo_10000.csv'
    output_file = 'results/evaluation_results_tokyo_10000.csv'

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Please run main.py first to generate geocoding results.")
        exit(1)

    # Create results directory if needed
    os.makedirs('results', exist_ok=True)

    # Run evaluation
    try:
        evaluate_geocoding_results(
            input_csv=input_file,
            output_csv=output_file,
            rate_limit_delay=0.1  # 100ms delay between API calls
        )

        print(f"\n{'='*80}")
        print("Evaluation Complete!")
        print(f"{'='*80}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
