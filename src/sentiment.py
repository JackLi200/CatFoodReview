# Apply VADER sentiment to cleaned review CSVs.

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Parse CLI arguments for sentiment labeling.
def parse_args():
    parser = argparse.ArgumentParser(description="Add sentiment scores/labels to cleaned reviews.")
    parser.add_argument("--inp", default="data/clean", help="Input directory with clean_*.csv")
    parser.add_argument("--out", default="data/clean", help="Output directory for with_sentiment_*.csv")
    return parser.parse_args()


# List cleaned files to process.
def list_clean_files(inp_dir: Path):
    return sorted(inp_dir.glob("clean_*.csv"))


# Apply VADER to one review body and return score/label.
def label_sentiment(text: str, analyzer: SentimentIntensityAnalyzer):
    scores = analyzer.polarity_scores(text or "")
    compound = scores["compound"]
    if compound >= 0.05:
        label = "pos"
    elif compound <= -0.05:
        label = "neg"
    else:
        label = "neu"
    return {"sentiment_score": compound, "sentiment_label": label}


# Load cleaned reviews, add sentiment columns, and write outputs.
def main():
    args = parse_args()
    inp_dir = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_clean_files(inp_dir)
    if not files:
        print(f"[warn] no clean_*.csv files found in {inp_dir}")
        return

    analyzer = SentimentIntensityAnalyzer()

    for path in files:
        df = pd.read_csv(path)
        if "review_body" not in df.columns:
            print(f"[warn] skipping {path.name}: missing review_body column")
            continue
        sentiments = df["review_body"].fillna("").astype(str).apply(lambda t: label_sentiment(t, analyzer))
        df["sentiment_score"] = sentiments.map(lambda d: d["sentiment_score"])
        df["sentiment_label"] = sentiments.map(lambda d: d["sentiment_label"])
        out_path = out_dir / path.name.replace("clean_", "with_sentiment_")
        df.to_csv(out_path, index=False)
        print(f"[ok] {path.name} -> {out_path.name} ({len(df)} rows)")


if __name__ == "__main__":
    main()
