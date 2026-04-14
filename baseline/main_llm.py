"""
LLM-only geocoding baseline.

Reads addresses from a CSV, calls LLM with address only, and outputs results
in the same format as main.py outputs.
"""

import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def parse_llm_response(response: str) -> Optional[Dict]:
    if not response:
        return None
    try:
        m = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        return None
    return None


def load_addresses(input_csv: str, column: str, limit: Optional[int]) -> List[str]:
    addresses = []
    with open(input_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if column not in fieldnames:
            column = fieldnames[0] if fieldnames else column
        for row in reader:
            value = row.get(column)
            if value:
                addresses.append(value.strip())
            if limit is not None and len(addresses) >= limit:
                break
    return addresses


def build_prompt(address: str) -> str:
    return f"""You are a professional Japanese geocoding expert. Please infer the most plausible coordinates based on the address text.

【Target Address】
{address}

Please return in JSON format:
{{
  "latitude": <latitude_value>,
  "longitude": <longitude_value>
}}

Return only JSON, no other text.
"""


def run_llm_only(input_csv: str, output_csv: str, column: str,
                 model: str, temperature: float, limit: Optional[int]):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    addresses = load_addresses(input_csv, column, limit)

    results = []
    success_count = 0
    failure_count = 0

    for address in tqdm(addresses, desc="Geocoding", unit="addr"):
        prompt = build_prompt(address)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional geocoding expert specializing in precise coordinate inference."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        parsed = parse_llm_response(content)

        if parsed and "latitude" in parsed and "longitude" in parsed:
            lat = float(parsed["latitude"])
            lon = float(parsed["longitude"])
        else:
            lat, lon = 36.0, 138.0

        if lat == 36.0 and lon == 138.0:
            failure_count += 1
        else:
            success_count += 1

        results.append({
            "address": address,
            "latitude": f"{lat:.6f}",
            "longitude": f"{lon:.6f}",
        })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "address",
            "latitude",
            "longitude",
        ])
        for r in results:
            writer.writerow([
                r["address"],
                r["latitude"],
                r["longitude"],
            ])

    print(f"Results saved to: {output_csv}")

    total = len(results)
    if total:
        print(f"\n{'='*80}")
        print("处理完成！")
        print(f"{'='*80}")
        print(f"总地址数: {total}")
        print(f"成功: {success_count}")
        print(f"失败: {failure_count}")
        print(f"成功率: {success_count/total*100:.1f}%")
        print(f"结果已保存到: {output_csv}")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="LLM-only geocoding baseline")
    parser.add_argument("--input", default="data_sample/test_random/tokyo_100.csv")
    parser.add_argument("--output", default="results/baseline/geocoding_results_llm_tokyo_100.csv")
    parser.add_argument("--column", default="place_name")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_llm_only(
        input_csv=args.input,
        output_csv=args.output,
        column=args.column,
        model=args.model,
        temperature=args.temperature,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
