# Generate simple visuals for sentiment distribution and top keywords.
# Outputs (default to data/outputs/figures/):
# - sentiment_dist.png : stacked bar (pos/neu/neg %) per product
# - rating_score.png   : bar chart of avg rating and sentiment score
# - top_terms_<product_id>_<bucket>.png : horizontal bars of top TF-IDF terms for pos/neg

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


# Parse CLI arguments for visualization.
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize sentiment and keywords.")
    parser.add_argument("--comparison", default="data/outputs/comparison.csv", help="Comparison CSV from aggregate.py")
    parser.add_argument("--keywords", default="data/outputs/keywords.csv", help="Keywords CSV from keywords.py")
    parser.add_argument("--out_dir", default="data/outputs/figures", help="Directory to save figures")
    parser.add_argument("--top_n", type=int, default=10, help="Top terms to show per bucket")
    return parser.parse_args()


# Stacked bar of pos/neu/neg percentages per product.
def plot_sentiment_dist(df: pd.DataFrame, out_path: Path):
    labels = df["display_name"].fillna(df["product_id"]).tolist()
    neg = df["neg_pct"].tolist()
    neu = df["neu_pct"].tolist()
    pos = df["pos_pct"].tolist()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, neg, label="neg", color="#d9544d")
    ax.bar(labels, neu, bottom=neg, label="neu", color="#c0c0c0")
    ax.bar(labels, pos, bottom=[n + u for n, u in zip(neg, neu)], label="pos", color="#5cb85c")
    ax.set_ylabel("Percentage")
    ax.set_title("Sentiment distribution by product")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# Dual-axis chart: average rating bars with sentiment score line.
def plot_rating_score(df: pd.DataFrame, out_path: Path):
    labels = df["display_name"].fillna(df["product_id"]).tolist()
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(labels, df["avg_rating"], color="#4a90e2", alpha=0.7, label="avg_rating")
    ax1.set_ylabel("Avg Rating (1-5)", color="#4a90e2")
    ax1.tick_params(axis="y", labelcolor="#4a90e2")
    plt.xticks(rotation=20, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(labels, df["score"], color="#f5a623", marker="o", label="sentiment score")
    ax2.set_ylabel("Sentiment Score (pos% - neg%)", color="#f5a623")
    ax2.tick_params(axis="y", labelcolor="#f5a623")

    plt.title("Ratings and sentiment score")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# Stacked bar of star-rating distribution (1-5★) per product.
def plot_rating_dist(df: pd.DataFrame, out_path: Path):
    labels = df["display_name"].fillna(df["product_id"]).tolist()
    r1, r2, r3, r4, r5 = (
        df["r1_pct"].tolist(),
        df["r2_pct"].tolist(),
        df["r3_pct"].tolist(),
        df["r4_pct"].tolist(),
        df["r5_pct"].tolist(),
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, r1, label="1★", color="#8b0000")
    ax.bar(labels, r2, bottom=r1, label="2★", color="#b22222")
    ax.bar(labels, r3, bottom=[a + b for a, b in zip(r1, r2)], label="3★", color="#cd853f")
    ax.bar(labels, r4, bottom=[a + b + c for a, b, c in zip(r1, r2, r3)], label="4★", color="#87cefa")
    ax.bar(labels, r5, bottom=[a + b + c + d for a, b, c, d in zip(r1, r2, r3, r4)], label="5★", color="#006400")
    ax.set_ylabel("Percentage")
    ax.set_title("Star rating distribution")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# Horizontal bar of top TF-IDF terms for one product/sentiment bucket.
def plot_top_terms(keywords_df: pd.DataFrame, product_id: str, product_label: str, bucket: str, top_n: int, out_path: Path):
    subset = keywords_df[(keywords_df["product_id"] == product_id) & (keywords_df["bucket"] == bucket)]
    subset = subset.sort_values("score", ascending=False).head(top_n)
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(subset["term"], subset["score"], color="#5cb85c" if bucket == "pos" else "#d9544d")
    ax.invert_yaxis()
    ax.set_title(f"Top {bucket} terms for {product_label}")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# Load comparison/keywords, render charts, and save figures.
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_path = Path(args.comparison)
    kw_path = Path(args.keywords)
    if not comp_path.exists():
        print(f"[warn] comparison file not found: {comp_path}")
        return
    comp_df = pd.read_csv(comp_path)
    if comp_df.empty:
        print("[warn] comparison file is empty")
        return

    plot_sentiment_dist(comp_df, out_dir / "sentiment_dist.png")
    plot_rating_score(comp_df, out_dir / "rating_score.png")
    if {"r1_pct", "r2_pct", "r3_pct", "r4_pct", "r5_pct"}.issubset(comp_df.columns):
        plot_rating_dist(comp_df, out_dir / "rating_dist.png")

    if kw_path.exists():
        kw_df = pd.read_csv(kw_path)
        for pid, label in comp_df[["product_id", "display_name"]].drop_duplicates().itertuples(index=False):
            for bucket in ["pos", "neg"]:
                out_path = out_dir / f"top_terms_{pid}_{bucket}.png"
                plot_top_terms(kw_df, pid, label, bucket, args.top_n, out_path)
    else:
        print(f"[warn] keywords file not found: {kw_path}")

    print(f"[ok] figures saved to {out_dir}")


if __name__ == "__main__":
    main()
