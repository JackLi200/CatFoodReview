import argparse
from pathlib import Path
from typing import List

import pandas as pd
from dateutil import parser as dateparser


def parse_args():
    parser = argparse.ArgumentParser(description="Clean raw review CSVs.")
    parser.add_argument("--inp", default="data/raw", help="Input directory with reviews_*.csv")
    parser.add_argument("--out", default="data/clean", help="Output directory for cleaned CSVs")
    parser.add_argument("--min_length", type=int, default=20, help="Minimum review body length to keep")
    return parser.parse_args()


# List raw review files to process.
def list_review_files(inp_dir: Path):
    return sorted(inp_dir.glob("reviews_*.csv"))


# Clean a single DataFrame of reviews.
def clean_frame(df: pd.DataFrame, min_length: int):
    df = df.copy()

    # Normalize text fields
    for col in ["review_body", "summary", "product_title"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # Drop empty or too-short reviews
    if "review_body" in df.columns:
        df = df[df["review_body"].str.len() >= min_length]

    # Coerce rating
    if "star_rating" in df.columns:
        df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")
        df = df[df["star_rating"].between(1, 5, inclusive="both")]

    # Parse dates into ISO
    if "review_date" in df.columns:
        df["review_date_iso"] = df["review_date"].apply(_safe_parse_date)

    # Drop duplicate review_ids then duplicate bodies
    if "review_id" in df.columns:
        df = df.drop_duplicates(subset=["review_id"])
    if "review_body" in df.columns:
        df = df.drop_duplicates(subset=["review_body"])

    return df.reset_index(drop=True)


# Safe date parse to ISO string.
def _safe_parse_date(val: str):
    try:
        dt = dateparser.parse(str(val))
        if dt:
            return dt.date().isoformat()
    except Exception:
        return ""
    return ""


def main():
    # Clean all reviews_*.csv files and write clean_*.csv outputs.
    args = parse_args()
    inp_dir = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_review_files(inp_dir)
    if not files:
        print(f"[warn] no reviews_*.csv files found in {inp_dir}")
        return

    for path in files:
        df = pd.read_csv(path)
        cleaned = clean_frame(df, min_length=args.min_length)
        out_path = out_dir / path.name.replace("reviews_", "clean_")
        cleaned.to_csv(out_path, index=False)
        print(f"[ok] {path.name} -> {out_path.name} ({len(cleaned)} rows)")


if __name__ == "__main__":
    main()
