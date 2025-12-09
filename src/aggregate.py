# Aggregate sentiment metrics and keywords into a comparison table with product metadata.

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


# Parse CLI arguments for aggregation.
def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate sentiment metrics across products.")
    parser.add_argument("--clean_dir", default="data/clean", help="Directory with with_sentiment_*.csv")
    parser.add_argument("--keywords", default="data/outputs/keywords.csv", help="CSV from keywords.py")
    parser.add_argument("--products", default="data/raw/products.csv", help="Products metadata CSV")
    parser.add_argument("--out", default="data/outputs", help="Output directory")
    return parser.parse_args()


# List sentiment files to process.
def list_files(clean_dir: Path):
    return sorted(clean_dir.glob("with_sentiment_*.csv"))


# Compute sentiment and rating distribution metrics for one product.
def sentiment_summary(df: pd.DataFrame):
    total = len(df)
    if total == 0:
        return {
            "total": 0,
            "pos_pct": 0,
            "neg_pct": 0,
            "neu_pct": 0,
            "avg_rating": 0,
            "verified_pct": 0,
            "avg_length": 0,
            "r1_pct": 0,
            "r2_pct": 0,
            "r3_pct": 0,
            "r4_pct": 0,
            "r5_pct": 0,
        }
    pos = (df["sentiment_label"] == "pos").sum()
    neg = (df["sentiment_label"] == "neg").sum()
    neu = (df["sentiment_label"] == "neu").sum()
    avg_rating = df["star_rating"].mean() if "star_rating" in df.columns else 0

    # Rating distribution
    ratings = df["star_rating"].fillna(0).astype(float).round().astype(int)
    dist = ratings.value_counts(normalize=True)
    r_pct = {f"r{k}_pct": float(dist.get(k, 0) * 100) for k in range(1, 6)}

    # Verified purchase pct (if column present)
    verified_pct = 0
    if "verified_purchase" in df.columns:
        verified = (
            df["verified_purchase"]
            .astype(str)
            .str.lower()
            .map(lambda x: x in {"true", "1", "yes"})
            .mean()
        )
        verified_pct = float(verified * 100)

    avg_length = df["review_body"].fillna("").astype(str).str.len().mean()

    return {
        "total": total,
        "pos_pct": pos / total * 100,
        "neg_pct": neg / total * 100,
        "neu_pct": neu / total * 100,
        "avg_rating": avg_rating,
        "verified_pct": verified_pct,
        "avg_length": avg_length,
        **r_pct,
    }


# Load keywords CSV (if present).
def load_keywords(path: Path):
    if not path.exists():
        return pd.DataFrame(columns=["product_id", "bucket", "term", "score"])
    return pd.read_csv(path)


# Load products metadata CSV (if present).
def load_products(path: Path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# Load per-product sentiment, merge metadata/keywords, compute scores, and write comparison files.
def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    keyword_df = load_keywords(Path(args.keywords))
    products_df = load_products(Path(args.products))
    meta = {row.product_id: row._asdict() for row in products_df.itertuples()} if not products_df.empty else {}

    rows = []
    summary_json: Dict[str, dict] = {}

    for path in list_files(Path(args.clean_dir)):
        df = pd.read_csv(path)
        pid = path.stem.replace("with_sentiment_", "")
        summ = sentiment_summary(df)

        # Simple score: positive - negative percentage
        score = summ["pos_pct"] - summ["neg_pct"]

        product_keywords = {}
        if not keyword_df.empty:
            subset = keyword_df[keyword_df["product_id"] == pid]
            for bucket in subset["bucket"].unique():
                terms = subset[subset["bucket"] == bucket].sort_values("score", ascending=False)
                product_keywords[bucket] = terms.head(10)[["term", "score"]].to_dict(orient="records")

        m = meta.get(pid, {})
        display_name = (m.get("brand") or "").strip() or pid

        row = {
            "product_id": pid,
            "display_name": display_name,
            "brand": m.get("brand"),
            "product_name": m.get("product_name"),
            "flavor": m.get("flavor"),
            "size_variant": m.get("size_variant"),
            "total_reviews": summ["total"],
            "pos_pct": summ["pos_pct"],
            "neg_pct": summ["neg_pct"],
            "neu_pct": summ["neu_pct"],
            "avg_rating": summ["avg_rating"],
            "verified_pct": summ["verified_pct"],
            "avg_length": summ["avg_length"],
            "r1_pct": summ["r1_pct"],
            "r2_pct": summ["r2_pct"],
            "r3_pct": summ["r3_pct"],
            "r4_pct": summ["r4_pct"],
            "r5_pct": summ["r5_pct"],
            "score": score,
            "top_pos_terms": product_keywords.get("pos", []),
            "top_neg_terms": product_keywords.get("neg", []),
        }
        rows.append(row)
        summary_json[pid] = {
            "meta": {
                "display_name": display_name,
                "brand": m.get("brand"),
                "product_name": m.get("product_name"),
                "flavor": m.get("flavor"),
                "size_variant": m.get("size_variant"),
            },
            "metrics": {k: v for k, v in summ.items()},
            "score": score,
            "keywords": product_keywords,
        }

    compare_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    compare_path = out_dir / "comparison.csv"
    compare_df.to_csv(compare_path, index=False)

    json_path = out_dir / "comparison.json"
    json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print(f"[ok] wrote comparison -> {compare_path} and {json_path}")


if __name__ == "__main__":
    main()
