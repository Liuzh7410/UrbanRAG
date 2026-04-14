import os
import pandas as pd
import time
from dotenv import load_dotenv
from openai import AzureOpenAI
import googlemaps
from math import radians, sin, cos, sqrt, atan2
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import requests

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Initialize Google Maps client
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_GEO_API_KEY"))

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
)
def get_gpt4_coordinates(address):
    """Get coordinates from GPT-4o model with retry logic"""
    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are a geocoding assistant. Given an address, return ONLY the latitude and longitude in the format: latitude,longitude. Do not include any other text or explanation."},
                {"role": "user", "content": f"What are the coordinates of this address: {address}"}
            ],
            temperature=0,
            timeout=30  # 添加超时设置
        )

        # Parse response
        coords_text = response.choices[0].message.content.strip()
        lat, lng = coords_text.split(',')
        return float(lat.strip()), float(lng.strip())
    except Exception as e:
        print(f"Error getting GPT-4 coordinates for {address}: {e}")
        return None, None

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
)
def get_google_coordinates(address):
    """Get coordinates from Google Maps API with retry logic"""
    try:
        geocode_result = gmaps.geocode(address)
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            return location['lat'], location['lng']
        return None, None
    except Exception as e:
        print(f"Error getting Google coordinates for {address}: {e}")
        return None, None

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in kilometers using Haversine formula"""
    if None in [lat1, lon1, lat2, lon2]:
        return None

    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    # Earth's radius in kilometers
    r = 6371

    return r * c

def main():
    # Read input data
    input_file = "data_sample/tokyo_sample100.csv"
    df = pd.read_csv(input_file)

    print(f"Processing {len(df)} addresses...")

    results = []

    for idx, row in df.iterrows():
        address = row['place_name']
        print(f"\nProcessing {idx + 1}/{len(df)}: {address}")

        # Get Google Maps coordinates (ground truth)
        true_lat, true_lng = get_google_coordinates(address)
        print(f"  Google Maps: ({true_lat}, {true_lng})")
        
        # 在两个API调用之间增加延迟
        time.sleep(1.5)

        # Get GPT-4 coordinates
        gpt4_lat, gpt4_lng = get_gpt4_coordinates(address)
        print(f"  GPT-4o: ({gpt4_lat}, {gpt4_lng})")

        # Calculate error
        error_km = calculate_distance(true_lat, true_lng, gpt4_lat, gpt4_lng)
        if error_km:
            print(f"  Error: {error_km:.4f} km")

        results.append({
            'place_name': address,
            'true_lat': true_lat,
            'true_lng': true_lng,
            'gpt4o_lat': gpt4_lat,
            'gpt4o_lng': gpt4_lng,
            'gpt4o_err_km': error_km
        })

        # 每10个地址保存一次中间结果
        if (idx + 1) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv('results/temp_progress.csv', index=False)
            print(f"  💾 Progress saved (processed {idx + 1}/{len(df)})")

        # 增加请求之间的延迟以避免限流
        time.sleep(2)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save results
    output_file = 'results/gpt4o_geocoding_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to {output_file}")

    # Print summary statistics
    valid_errors = results_df['gpt4o_err_km'].dropna()
    if len(valid_errors) > 0:
        print(f"\n=== Summary Statistics ===")
        print(f"Total addresses: {len(results_df)}")
        print(f"Successful geocoding: {len(valid_errors)}")
        print(f"Mean error: {valid_errors.mean():.4f} km")
        print(f"Median error: {valid_errors.median():.4f} km")
        print(f"Min error: {valid_errors.min():.4f} km")
        print(f"Max error: {valid_errors.max():.4f} km")
        print(f"Std deviation: {valid_errors.std():.4f} km")

if __name__ == "__main__":
    main()
