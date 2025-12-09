# Extract top keywords/phrases per product and sentiment bucket using TF-IDF.

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    parser = argparse.ArgumentParser(description="Extract TF-IDF keywords per product.")
    parser.add_argument("--inp", default="data/clean", help="Directory with with_sentiment_*.csv")
    parser.add_argument("--out", default="data/outputs", help="Directory to write keywords JSON/CSV")
    parser.add_argument("--per_sentiment", action="store_true", default=True, help="Split by sentiment buckets")
    parser.add_argument("--products", default="data/raw/products.csv", help="Products CSV (used to strip brand tokens)")
    parser.add_argument("--min_df", type=int, default=2, help="Minimum doc frequency")
    parser.add_argument("--max_features", type=int, default=2000, help="Max TF-IDF features")
    parser.add_argument("--top_k", type=int, default=20, help="Top terms to keep per bucket")
    return parser.parse_args()


def list_files(inp: Path):
    return sorted(inp.glob("with_sentiment_*.csv"))


def build_vectorizer(min_df: int, max_features: int):
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        lowercase=True,
    )


EXTRA_STOP = {
    "cat",
    "cats",
    "kitty",
    "kitten",
    "kittens",
    "feline",
    "pet",
    "pets",
    "food",
    "bag",
    "bags",
    "pound",
    "pounds",
    "lb",
    "lbs",
    "ounce",
    "ounces",
    "like",
    "buy",
    "bought",
    "purchase",
    "product",
    "brand",
}


def _is_noise(term: str, extra_stop: Set[str]):
    tokens = term.split()
    return any(tok in extra_stop for tok in tokens)


def extract_top_terms(texts: List[str], top_k: int, vectorizer: TfidfVectorizer, stop: Set[str]):
    if not texts:
        return []
    X = vectorizer.fit_transform(texts)
    scores = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    paired = list(zip(terms, scores))
    paired = [(t, s) for t, s in paired if not _is_noise(t, stop)]
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired[:top_k]


def load_brand_stop(products_path: Path):
    # Build a stoplist from static noise terms plus brand/product tokens.
    stop: Set[str] = set(EXTRA_STOP)
    if not products_path.exists():
        return stop
    df = pd.read_csv(products_path)
    for col in ["brand", "product_name"]:
        if col in df.columns:
            for val in df[col].dropna().astype(str):
                for tok in val.lower().split():
                    stop.add(tok.strip(",.'\""))
    return stop


def process_file(path: Path, args: argparse.Namespace, stop: Set[str]):
    df = pd.read_csv(path)
    if "review_body" not in df.columns:
        return {}
    product_id = path.stem.replace("with_sentiment_", "")

    buckets: Dict[str, List[str]] = defaultdict(list)
    if args.per_sentiment and "sentiment_label" in df.columns:
        for label in ["pos", "neg", "neu"]:
            subset = df[df["sentiment_label"] == label]
            buckets[label] = subset["review_body"].dropna().astype(str).tolist()
    else:
        buckets["all"] = df["review_body"].dropna().astype(str).tolist()

    results: Dict[str, List[Tuple[str, float]]] = {}
    for bucket, texts in buckets.items():
        vec = build_vectorizer(args.min_df, args.max_features)
        results[bucket] = extract_top_terms(texts, args.top_k, vec, stop)
    return {product_id: results}


def main():
    # Extract top terms per product/bucket and write JSON/CSV artifacts.
    args = parse_args()
    inp_dir = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stop = load_brand_stop(Path(args.products))

    records = {}
    csv_rows = []
    for path in list_files(inp_dir):
        res = process_file(path, args, stop)
        if not res:
            continue
        product_id = next(iter(res))
        records[product_id] = res[product_id]
        for bucket, terms in res[product_id].items():
            for term, score in terms:
                csv_rows.append({"product_id": product_id, "bucket": bucket, "term": term, "score": score})

    json_path = out_dir / "keywords.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    csv_path = out_dir / "keywords.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"[ok] wrote keywords -> {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
