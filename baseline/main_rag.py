"""
Plain text RAG baseline for geocoding.

Build embeddings for the plaintext corpus (if needed), retrieve top-k entries,
then ask the LLM to infer coordinates.
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
import statistics


DEFAULT_CONFIGS = {
    "rag": {
        "corpus": "corpus/RAG_tokyo_corpus.jsonl",
        "embeddings": "corpus/embeddings/RAG_tokyo_corpus_embeddings.jsonl",
        "output": "results/baseline/geocoding_results_rag_tokyo_10000.csv",
    },
    "lightrag": {
        "corpus": "corpus/lightrag_tokyo_corpus.jsonl",
        "embeddings": "corpus/embeddings/LightRAG_tokyo_corpus_embeddings.jsonl",
        "output": "results/baseline/geocoding_results_lightrag_tokyo_10000.csv",
    },
    "graphrag": {
        "corpus": "corpus/lightrag_tokyo_corpus.jsonl",
        "embeddings": "corpus/embeddings/LightRAG_tokyo_corpus_embeddings.jsonl",
        "output": "results/baseline/geocoding_results_graphrag_tokyo_10000.csv",
    },
}


def parse_point_wkt(wkt_str: Optional[str]) -> Optional[Tuple[float, float]]:
    if not wkt_str:
        return None
    m = re.match(r"^POINT\s*\(([-\d\.]+)\s+([-\d\.]+)\)$", str(wkt_str).strip())
    if not m:
        return None
    lon = float(m.group(1))
    lat = float(m.group(2))
    return lat, lon


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    print(f"Loading JSONL: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Load {os.path.basename(path)}", unit="line"):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"Loaded records: {len(rows)}")
    return rows


def sanitize_text_for_api(text: str, max_chars: int = 4000) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text[:max_chars]
    text = text.replace("\x00", " ")
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\r\t")
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return text


def write_jsonl(path: str, rows: Iterable[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def embed_texts(client: OpenAI, model: str, texts: List[str], max_chars: int = 4000) -> List[List[float]]:
    if not texts:
        return []
    safe_texts = [sanitize_text_for_api(t, max_chars=max_chars) for t in texts]
    resp = client.embeddings.create(model=model, input=safe_texts)
    return [d.embedding for d in resp.data]


def sanitize_text_for_query(text: str, max_chars: int = 512) -> str:
    text = sanitize_text_for_api(text, max_chars=max_chars)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_embed_query(client: OpenAI, model: str, query_text: str) -> Optional[List[float]]:
    candidates = [
        sanitize_text_for_query(query_text, max_chars=512),
        sanitize_text_for_query(query_text, max_chars=256),
        sanitize_text_for_query(query_text, max_chars=128),
    ]
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return embed_texts(client, model, [candidate], max_chars=len(candidate))[0]
        except Exception as exc:
            print(f"\nQuery embedding retry failed for address variant={candidate!r} error={exc}")
            continue
    return None


def embed_batch_resilient(
    client: OpenAI,
    model: str,
    batch: List[Dict],
    batch_start: int,
    max_chars: int = 4000,
) -> List[Dict]:
    if not batch:
        return []

    texts = [row.get("text", "") for row in batch]
    try:
        embeddings = embed_texts(client, model, texts, max_chars=max_chars)
        return [
            {
                "type": row.get("type"),
                "id": row.get("id"),
                "text": row.get("text"),
                "meta": row.get("meta", {}),
                "embedding": emb,
            }
            for row, emb in zip(batch, embeddings)
        ]
    except Exception as exc:
        if len(batch) == 1:
            row = batch[0]
            print(
                f"\nSkip bad embedding row at corpus index {batch_start}: "
                f"id={row.get('id')} type={row.get('type')} error={exc}"
            )
            return [{
                "type": row.get("type"),
                "id": row.get("id"),
                "text": row.get("text"),
                "meta": row.get("meta", {}),
                "embedding": None,
                "embedding_error": str(exc),
            }]

        mid = len(batch) // 2
        left = embed_batch_resilient(client, model, batch[:mid], batch_start, max_chars=max_chars)
        right = embed_batch_resilient(client, model, batch[mid:], batch_start + mid, max_chars=max_chars)
        return left + right


def append_jsonl(path: str, rows: Iterable[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_embeddings(client: OpenAI, corpus: List[Dict], model: str,
                     batch_size: int, output_path: str, existing_rows: Optional[List[Dict]] = None) -> List[Dict]:
    rows = list(existing_rows or [])
    start_idx = len(rows)
    total_batches = (len(corpus) - start_idx + batch_size - 1) // batch_size if len(corpus) > start_idx else 0
    for i in tqdm(
        range(start_idx, len(corpus), batch_size),
        total=total_batches,
        desc=f"Building embeddings {os.path.basename(output_path)}",
        unit="batch",
    ):
        batch = corpus[i:i + batch_size]
        batch_rows = embed_batch_resilient(client, model, batch, i)
        append_jsonl(output_path, batch_rows)
        rows.extend(batch_rows)
    return rows


def load_or_build_embeddings(client: OpenAI, corpus_path: str, embed_path: str,
                             model: str, batch_size: int, rebuild: bool) -> List[Dict]:
    if os.path.exists(corpus_path):
        corpus_size_mb = os.path.getsize(corpus_path) / (1024 * 1024)
        print(f"Corpus file: {corpus_path} ({corpus_size_mb:.1f} MB)")
    print("Loading corpus...")
    corpus = load_jsonl(corpus_path)
    print(f"Corpus records: {len(corpus)}")

    if os.path.exists(embed_path) and not rebuild:
        size_mb = os.path.getsize(embed_path) / (1024 * 1024)
        existing_rows = load_jsonl(embed_path)
        if len(existing_rows) >= len(corpus):
            print(f"Using existing embeddings: {embed_path} ({size_mb:.1f} MB)")
            return existing_rows
        print(
            f"Existing embeddings are incomplete: {len(existing_rows)}/{len(corpus)} rows. "
            f"Resume building from partial file."
        )
        return build_embeddings(client, corpus, model, batch_size, embed_path, existing_rows=existing_rows)

    print("Building embeddings from corpus...")
    existing_rows: List[Dict] = []
    if os.path.exists(embed_path) and rebuild:
        print(f"Rebuild requested, overwriting existing embeddings: {embed_path}")
        os.remove(embed_path)
    elif os.path.exists(embed_path):
        existing_rows = load_jsonl(embed_path)
        print(f"Resume from existing embeddings: {len(existing_rows)} rows")
    return build_embeddings(client, corpus, model, batch_size, embed_path, existing_rows=existing_rows)


def build_embedding_matrix(indexed: List[Dict]) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    valid_rows = []
    vectors = []
    for row in indexed:
        emb = row.get("embedding")
        if not emb:
            continue
        valid_rows.append(row)
        vectors.append(emb)
    if not vectors:
        return valid_rows, np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0.0] = 1.0
    return valid_rows, matrix, norms


def parse_lightrag_triple_row(row: Dict) -> Optional[Tuple[str, str, str]]:
    text = row.get("text", "")
    match = re.search(r'([A-Za-z_]+)\((.*?)\)\s*-\[([A-Za-z_]+)\]->\s*([A-Za-z_]+)\((.*?)\)', text)
    if not match:
        return None
    _, head_name, relation, _, tail_name = match.groups()
    return head_name, relation, tail_name


def extract_entities_from_lightrag_row(row: Dict) -> List[str]:
    entities = []
    meta = row.get("meta", {}) or {}
    if row.get("type") == "node":
        for key in ("name", "address", "block_address", "area_name", "ward_name"):
            if meta.get(key):
                entities.append(str(meta[key]))
    elif row.get("type") == "triple":
        triple = parse_lightrag_triple_row(row)
        if triple:
            entities.extend([triple[0], triple[2]])
    return list(dict.fromkeys(entities))


def build_lightrag_graph_index(indexed_rows: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    entity_to_rows: Dict[str, List[Dict]] = defaultdict(list)
    entity_to_triples: Dict[str, List[Dict]] = defaultdict(list)
    for row in indexed_rows:
        entities = extract_entities_from_lightrag_row(row)
        for entity in entities:
            entity_to_rows[entity].append(row)
            if row.get("type") == "triple":
                entity_to_triples[entity].append(row)
    return {
        "entity_to_rows": entity_to_rows,
        "entity_to_triples": entity_to_triples,
    }


def retrieve_top_k(query_emb: List[float], indexed_rows: List[Dict], matrix: np.ndarray,
                   norms: np.ndarray, k: int) -> List[Dict]:
    if matrix.size == 0:
        return []
    query = np.asarray(query_emb, dtype=np.float32)
    qnorm = np.linalg.norm(query)
    if qnorm == 0.0:
        qnorm = 1.0
    sims = (matrix @ query) / (norms * qnorm)
    top_idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [indexed_rows[i] for i in top_idx]


def retrieve_graph_expanded_contexts(query_emb: List[float], indexed_rows: List[Dict], matrix: np.ndarray,
                                     norms: np.ndarray, k: int,
                                     graph_index: Dict[str, Dict[str, List[Dict]]]) -> List[Dict]:
    seed_rows = retrieve_top_k(query_emb, indexed_rows, matrix, norms, k)
    seed_entities: List[str] = []
    for row in seed_rows:
        seed_entities.extend(extract_entities_from_lightrag_row(row))
    seed_entities = list(dict.fromkeys(seed_entities))[:20]

    expanded_rows: List[Dict] = []
    seen_ids = set()
    for row in seed_rows:
        row_id = str(row.get("id"))
        if row_id not in seen_ids:
            expanded_rows.append(row)
            seen_ids.add(row_id)

    for entity in seed_entities:
        for row in graph_index["entity_to_triples"].get(entity, [])[:12]:
            row_id = str(row.get("id"))
            if row_id not in seen_ids:
                expanded_rows.append(row)
                seen_ids.add(row_id)
        for row in graph_index["entity_to_rows"].get(entity, [])[:6]:
            row_id = str(row.get("id"))
            if row_id not in seen_ids:
                expanded_rows.append(row)
                seen_ids.add(row_id)
        if len(expanded_rows) >= k * 8:
            break
    return expanded_rows


def build_prompt(address: str, contexts: List[Dict], max_context_chars: int) -> str:
    prompt = [
        "You are a geocoding expert.",
        "Infer the most plausible coordinates using the retrieved context.",
        "Return JSON only.",
        "",
        "Target Address:",
        sanitize_text_for_api(address, max_chars=max_context_chars),
        "",
        "Retrieved Context:",
    ]

    for i, c in enumerate(contexts, 1):
        text = sanitize_text_for_api(c.get("text", ""), max_chars=max_context_chars)
        if len(text) > max_context_chars:
            text = text[:max_context_chars] + "..."
        prompt.append(f"{i}. {text}")

    prompt.extend([
        "",
        "Return JSON:",
        "{",
        "  \"latitude\": <latitude_value>,",
        "  \"longitude\": <longitude_value>,",
        "  \"reasoning\": \"<brief reasoning>\",",
        "  \"confidence\": \"low\"",
        "}",
    ])
    return sanitize_text_for_api("\n".join(prompt), max_chars=max(max_context_chars * max(1, len(contexts) + 8), 4000))


def safe_chat_completion(
    client: OpenAI,
    llm_model: str,
    prompt: str,
    max_tokens: int,
) -> str:
    try:
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a geocoding expert."},
                {"role": "user", "content": sanitize_text_for_api(prompt, max_chars=16000)},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        compact_prompt = sanitize_text_for_api(prompt, max_chars=4000)
        print(f"\nChat request failed once; retry with compact sanitized prompt. error={exc}")
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a geocoding expert."},
                {"role": "user", "content": compact_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


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


def average_coords_from_context(contexts: List[Dict]) -> Optional[Tuple[float, float]]:
    coords = []
    for c in contexts:
        meta = c.get("meta", {})
        geom = meta.get("geometry")
        latlon = parse_point_wkt(geom)
        if latlon:
            coords.append(latlon)
    if not coords:
        return None
    avg_lat = sum(p[0] for p in coords) / len(coords)
    avg_lon = sum(p[1] for p in coords) / len(coords)
    return avg_lat, avg_lon


def load_existing_progress(output_csv: str) -> Tuple[int, int, int]:
    if not os.path.exists(output_csv):
        return 0, 0, 0
    total = 0
    success = 0
    failure = 0
    with open(output_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            conf = row.get("confidence", "")
            if conf in ["failed", "very_low"]:
                failure += 1
            else:
                success += 1
    return total, success, failure


def run_rag(input_csv: str, output_csv: str, corpus_path: str, embed_path: str,
            top_k: int, embed_model: str, llm_model: str, batch_size: int,
            rebuild_embeddings: bool, limit: Optional[int], column: str,
            max_context_chars: int, max_tokens: int, resume: bool,
            save_every: int, method: str):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    print("Loading/building embeddings...")

    indexed = load_or_build_embeddings(client, corpus_path, embed_path,
                                       embed_model, batch_size, rebuild_embeddings)
    indexed_rows, emb_matrix, emb_norms = build_embedding_matrix(indexed)
    print(f"Embeddings ready: {len(indexed_rows)} records")
    graph_index = build_lightrag_graph_index(indexed_rows) if method == "graphrag" else None

    rows = []
    with open(input_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if column not in fieldnames:
            column = fieldnames[0] if fieldnames else column
        for row in reader:
            address = row.get(column) or row.get("address") or row.get("Address") or row.get("住所")
            if address:
                rows.append(address.strip())
    if limit is not None and len(rows) > limit:
        rows = rows[:limit]
        print(f"Using first {limit} addresses (sequential)")
    print(f"Loaded addresses: {len(rows)}")

    start_idx = 0
    success_count = 0
    failure_count = 0
    if resume:
        done_count, done_success, done_failure = load_existing_progress(output_csv)
        start_idx = min(done_count, len(rows))
        success_count += done_success
        failure_count += done_failure
        print(f"Resume enabled: skip {start_idx} already processed rows")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    file_exists = os.path.exists(output_csv)
    mode = "a" if (resume and file_exists) else "w"
    with open(output_csv, mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow([
                "address",
                "latitude",
                "longitude",
                "confidence",
                "scenario",
                "method",
                "reasoning",
                "metadata",
            ])

        for idx, address in enumerate(tqdm(rows[start_idx:], desc="Geocoding", unit="addr"), start_idx + 1):
            query_emb = safe_embed_query(client, embed_model, address)
            if query_emb is None:
                contexts = []
                reasoning_prefix = "Query embedding failed; "
            elif method == "graphrag":
                contexts = retrieve_graph_expanded_contexts(
                    query_emb, indexed_rows, emb_matrix, emb_norms, top_k, graph_index
                )
                reasoning_prefix = ""
            else:
                contexts = retrieve_top_k(query_emb, indexed_rows, emb_matrix, emb_norms, top_k)
                reasoning_prefix = ""

            prompt = build_prompt(address, contexts, max_context_chars)
            content = safe_chat_completion(client, llm_model, prompt, max_tokens)
            parsed = parse_llm_response(content)

            if parsed and parsed.get("latitude") is not None and parsed.get("longitude") is not None:
                lat = float(parsed["latitude"])
                lon = float(parsed["longitude"])
                confidence = parsed.get("confidence", "low")
                reasoning = parsed.get("reasoning", "RAG inference")
            else:
                avg = average_coords_from_context(contexts)
                if avg:
                    lat, lon = avg
                    confidence = "very_low"
                    reasoning = reasoning_prefix + "LLM parse failed, using average of retrieved coordinates"
                else:
                    lat, lon = 36.0, 138.0
                    confidence = "very_low"
                    reasoning = reasoning_prefix + "LLM parse failed, using rough Japan centroid"

            writer.writerow([
                address,
                f"{lat:.6f}",
                f"{lon:.6f}",
                confidence,
                (
                    "graph_rag" if method == "graphrag"
                    else ("light_rag" if method == "lightrag" else "plaintext_rag")
                ),
                (
                    "GraphRAG_lightrag" if method == "graphrag"
                    else ("LightRAG" if method == "lightrag" else "RAG_plaintext")
                ),
                reasoning,
                json.dumps({"top_k": top_k, "method": method}, ensure_ascii=False),
            ])

            if confidence in ["failed", "very_low"]:
                failure_count += 1
            else:
                success_count += 1

            if idx % 10 == 0:
                tqdm.write(f"Processed {idx}/{len(rows)}")
            if idx % save_every == 0:
                f.flush()

    print(f"Results saved to: {output_csv}")

    total = success_count + failure_count
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
    parser = argparse.ArgumentParser(description="Plain text RAG geocoding baseline")
    parser.add_argument("--method", choices=["rag", "lightrag", "graphrag"], default="rag")
    parser.add_argument("--input", default="data_sample/test_random/tokyo_10000.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--embed-model", default="text-embedding-3-small")
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--column", default="place_name")
    parser.add_argument("--max-context-chars", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    method_defaults = DEFAULT_CONFIGS[args.method]
    corpus_path = args.corpus or method_defaults["corpus"]
    embed_path = args.embeddings or method_defaults["embeddings"]
    output_path = args.output or method_defaults["output"]

    run_rag(
        input_csv=args.input,
        output_csv=output_path,
        corpus_path=corpus_path,
        embed_path=embed_path,
        top_k=args.top_k,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        batch_size=args.batch_size,
        rebuild_embeddings=args.rebuild_embeddings,
        limit=args.limit,
        column=args.column,
        max_context_chars=args.max_context_chars,
        max_tokens=args.max_tokens,
        resume=args.resume,
        save_every=args.save_every,
        method=args.method,
    )


if __name__ == "__main__":
    main()
