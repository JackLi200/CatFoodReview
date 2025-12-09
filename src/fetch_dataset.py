# Fetch cat food product reviews from the open Amazon Pet Supplies 5-core dataset
# (`Pet_Supplies_5.json.gz`) hosted by UCSD.
# This script matches reviews to products using either ASINs (if provided) or
# manual phrase matching on product titles.

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests


DATA_URL = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Pet_Supplies_5.json.gz"
DATA_FILENAME = "Pet_Supplies_5.json.gz"

# Manual phrases tuned to reduce false positives (used only if ASINs unavailable).
PRESET_PHRASES: Dict[str, List[str]] = {
    "p1": ["purina one tender selects", "tender selects", "purina one"],
    "p2": ["blue buffalo wilderness", "wilderness high protein", "blue buffalo high protein"],
    "p3": ["science diet indoor", "hill s science diet indoor", "hill science diet indoor"],
    "p4": ["iams indoor weight", "iams hairball", "proactive health indoor"],
    "p5": ["royal canin indoor", "feline care nutrition indoor"],
}


# Build phrase lists per product from metadata (brand/name/flavor) or presets.
def build_phrase_map(products_csv: Path):
    df = pd.read_csv(products_csv)
    phrases = {}
    for _, row in df.iterrows():
        pid = str(row["product_id"])
        if pid in PRESET_PHRASES:
            phrases[pid] = PRESET_PHRASES[pid]
            continue
        brand = normalize(str(row.get("brand", "")))
        name = normalize(str(row.get("product_name", "")))
        flavor = normalize(str(row.get("flavor", "")))
        candidates = []
        for frag in [name, flavor]:
            if frag:
                candidates.append(frag)
                parts = frag.split()
                if len(parts) > 3:
                    candidates.append(" ".join(parts[:3]))
        if brand:
            candidates.append(brand)
        phrases[pid] = [c for c in candidates if c]
    return phrases


# Return True if any phrase is contained in the normalized title.
def matches(title_norm: str, phrase_list: Iterable[str]):
    return any(p in title_norm for p in phrase_list)


# Download the UCSD 5-core Pet Supplies file if it is not already present.
def ensure_dataset_file(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / DATA_FILENAME
    if dest.exists():
        return dest
    print(f"[info] downloading dataset to {dest}")
    with requests.get(DATA_URL, stream=True, timeout=30, verify=False) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"[ok] downloaded {dest}")
    return dest


# Stream the dataset, match by ASIN (preferred) or phrases, and write per-product CSVs.
def fetch_reviews(
    products_csv: Path,
    out_dir: Path,
    max_per_product: int,
    max_scan: int,
    review_urls_csv: Optional[Path] = None,
):
    phrase_map = build_phrase_map(products_csv)
    asin_map: Dict[str, str] = {}
    if review_urls_csv and review_urls_csv.exists():
        df_asin = pd.read_csv(review_urls_csv)
        if not {"product_id", "asin"} <= set(df_asin.columns):
            print(f"[warn] review_urls file missing required columns, falling back to phrase match: {review_urls_csv}")
        else:
            asin_map = {str(r.product_id): str(r.asin).strip().upper() for r in df_asin.itertuples()}
    asin_to_pid = {asin: pid for pid, asin in asin_map.items() if asin}

    targets = asin_map.keys() if asin_map else phrase_map.keys()
    counts = {pid: 0 for pid in targets}
    rows: Dict[str, List[dict]] = {pid: [] for pid in targets}

    dataset_path = ensure_dataset_file(out_dir)

    scanned = 0
    try:
        with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
            for line in f:
                scanned += 1
                ex = json.loads(line)
                asin = str(ex.get("asin", "")).upper()
                matched_pid = asin_to_pid.get(asin)

                # Primary path: ASIN match
                if matched_pid:
                    if counts[matched_pid] < max_per_product:
                        rows[matched_pid].append(
                            {
                                "product_id": matched_pid,
                                "asin": asin,
                                "review_id": ex.get("reviewerID"),
                                "product_title": None,
                                "star_rating": ex.get("overall"),
                                "review_body": ex.get("reviewText"),
                                "review_date": ex.get("reviewTime"),
                                "summary": ex.get("summary"),
                                "verified_purchase": ex.get("verified", None),
                            }
                        )
                        counts[matched_pid] += 1

                # Fallback: phrase match on title (if available) when no ASIN map
                elif not asin_map:
                    title_norm = normalize(ex.get("title", ""))
                    if not title_norm:
                        continue
                    for pid, phrases in phrase_map.items():
                        if counts[pid] >= max_per_product:
                            continue
                        if matches(title_norm, phrases):
                            rows[pid].append(
                                {
                                    "product_id": pid,
                                    "asin": asin,
                                    "review_id": ex.get("reviewerID"),
                                    "product_title": ex.get("title"),
                                    "star_rating": ex.get("overall"),
                                    "review_body": ex.get("reviewText"),
                                    "review_date": ex.get("reviewTime"),
                                    "summary": ex.get("summary"),
                                    "verified_purchase": ex.get("verified", None),
                                }
                            )
                            counts[pid] += 1
                if scanned % 100000 == 0:
                    print(f"[info] scanned {scanned}, counts={counts}")
                if scanned >= max_scan or all(counts[pid] >= max_per_product for pid in counts):
                    break
    except Exception as exc: 
        print(f"[error] Failed during read/parse: {exc}", file=sys.stderr)
        return

    for pid, data in rows.items():
        if not data:
            print(f"[warn] No reviews collected for {pid}.")
            continue
        out_path = out_dir / f"reviews_{pid}.csv"
        pd.DataFrame(data).to_csv(out_path, index=False)
        print(f"[ok] {pid}: wrote {len(data)} reviews -> {out_path}")

    print(f"[done] scanned={scanned}, counts={counts}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch reviews from UCSD Amazon Pet Supplies 5-core dataset.")
    parser.add_argument("--products", default="data/raw/products.csv", help="Products CSV path.")
    parser.add_argument("--out_dir", default="data/raw/", help="Directory for output review CSVs (also holds dataset file).")
    parser.add_argument(
        "--max_per_product",
        type=int,
        default=500,
        help="Max reviews to collect per product.",
    )
    parser.add_argument(
        "--max_scan",
        type=int,
        default=800000,
        help="Max lines to scan before stopping.",
    )
    parser.add_argument(
        "--review_urls",
        default="data/raw/review_urls.csv",
        help="CSV with product_id,asin columns (used for direct ASIN filtering).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fetch_reviews(
        products_csv=Path(args.products),
        out_dir=Path(args.out_dir),
        max_per_product=args.max_per_product,
        max_scan=args.max_scan,
        review_urls_csv=Path(args.review_urls) if args.review_urls else None,
    )


if __name__ == "__main__":
    main()
