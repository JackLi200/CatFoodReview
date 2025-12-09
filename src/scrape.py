# Review scraper for cat food products.
# Supports:
# 1) Amazon review pages (preferred: predictable structure, fewer bot blocks).
#    - If the URL has an ASIN (/dp/<asin>), it builds the product-review URL.
#    - Otherwise, with --amazon-search (default), it searches Amazon for the first ASIN
#      using brand/product name/flavor/size, then scrapes reviews.
#    - Parses review cards (div[data-hook="review"]) directly from HTML.
# 2) Fallback JSON-LD parsing for pages exposing <script type="application/ld+json"> reviews
#    (works for some sites if you save HTML manually via --html_dir).

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape reviews from product pages.")
    parser.add_argument(
        "--products",
        default="data/raw/products.csv",
        help="Path to products CSV with columns including product_id and source_url.",
    )
    parser.add_argument(
        "--out_dir",
        default="data/raw/",
        help="Directory to write per-product review CSVs.",
    )
    parser.add_argument(
        "--html_dir",
        default=None,
        help="Optional directory containing pre-downloaded HTML files named <product_id>.html to parse instead of fetching.",
    )
    parser.add_argument(
        "--save_html_dir",
        default=None,
        help="If set, save fetched HTML to this directory for inspection.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Seconds to sleep between requests to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--max_reviews",
        type=int,
        default=400,
        help="Max reviews to keep per product (after parsing).",
    )
    parser.add_argument(
        "--amazon_pages",
        type=int,
        default=3,
        help="Max Amazon review pages to fetch (roughly 10 reviews/page).",
    )
    parser.add_argument(
        "--amazon_search",
        action="store_true",
        default=True,
        help="Enable searching Amazon for an ASIN when URL is non-Amazon or missing an ASIN.",
    )
    parser.add_argument(
        "--no-amazon-search",
        dest="amazon_search",
        action="store_false",
        help="Disable Amazon search fallback.",
    )
    parser.add_argument(
        "--force_amazon",
        action="store_true",
        help="When source is non-Amazon, try to find an Amazon ASIN and scrape Amazon reviews instead.",
    )
    return parser.parse_args()


def load_products(path: Path):
    # Read products CSV and validate required columns.
    df = pd.read_csv(path)
    required_cols = {"product_id", "brand", "product_name", "source_site", "source_url"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in products CSV: {missing}")
    return df


def fetch_html(url: str, timeout: int):
    # Simple HTTP GET with headers to mimic a browser.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed to fetch {url}: {exc}")
        return None


def _flatten_json_ld(data: Any):
    """Flatten JSON-LD blocks to a list of dicts."""
    items: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        if data.get("@type") or data.get("@graph"):
            items.append(data)
        if "@graph" in data and isinstance(data["@graph"], list):
            for obj in data["@graph"]:
                if isinstance(obj, dict):
                    items.append(obj)
    elif isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                items.append(obj)
    return items


def extract_jsonld_reviews(html: str):
    """Parse review objects from <script type='application/ld+json'> blocks."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: List[Dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
        except Exception:
            continue
        for obj in _flatten_json_ld(data):
            if not isinstance(obj, dict):
                continue
            if obj.get("@type") in ("Review", ["Review"]):
                reviews.append(obj)
            if "review" in obj and isinstance(obj["review"], list):
                for nested in obj["review"]:
                    if isinstance(nested, dict) and nested.get("@type") == "Review":
                        reviews.append(nested)
    return reviews


# --- Amazon-specific helpers ---
def parse_asin_from_url(url: str):
    """Extract ASIN from common Amazon URL patterns."""
    import re

    patterns = [
        r"/dp/([A-Z0-9]{10})",
        r"/gp/product/([A-Z0-9]{10})",
        r"/product-reviews/([A-Z0-9]{10})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def search_amazon_for_asin(query: str, timeout: int):
    """Search Amazon and return the first ASIN."""
    from urllib.parse import quote_plus

    search_url = f"https://www.amazon.com/s?k={quote_plus(query)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(search_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Amazon search failed for '{query}': {exc}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    result = soup.find("div", attrs={"data-asin": True, "data-component-type": "s-search-result"})
    if result:
        asin = result.get("data-asin")
        if asin and len(asin) == 10:
            return asin
    print(f"[warn] Amazon search found no ASIN for '{query}'")
    return None


def build_amazon_reviews_url(meta: Dict[str, Any], timeout: int, allow_search: bool):
    """Return (reviews_url, asin) or None."""
    asin = None
    if isinstance(meta.get("source_url"), str):
        asin = parse_asin_from_url(meta["source_url"])
    if not asin and allow_search:
        q_parts = [
            str(meta.get("brand", "")),
            str(meta.get("product_name", "")),
            str(meta.get("flavor", "")),
            str(meta.get("size_variant", "")),
        ]
        query = " ".join(p for p in q_parts if p)
        asin = search_amazon_for_asin(query, timeout=timeout)
    if not asin:
        return None
    url = f"https://www.amazon.com/product-reviews/{asin}?reviewerType=all_reviews"
    return url, asin


def parse_amazon_reviews(html: str):
    """Parse reviews from an Amazon product-reviews page."""
    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.find_all("div", attrs={"data-hook": "review"})
    reviews: List[Dict[str, Any]] = []
    for blk in blocks:
        review_id = blk.get("id")
        title_el = blk.find(attrs={"data-hook": "review-title"})
        title = title_el.get_text(strip=True) if title_el else None
        body_el = blk.find(attrs={"data-hook": "review-body"})
        body = body_el.get_text(" ", strip=True) if body_el else None
        rating_el = blk.find(attrs={"data-hook": "review-star-rating"})
        rating = None
        if rating_el:
            txt = rating_el.get_text(strip=True)
            try:
                rating = float(txt.split()[0])  # "5.0 out of 5 stars"
            except Exception:
                rating = txt
        author_el = blk.find("span", class_="a-profile-name")
        author = author_el.get_text(strip=True) if author_el else None
        date_el = blk.find(attrs={"data-hook": "review-date"})
        date = date_el.get_text(strip=True) if date_el else None

        reviews.append(
            {
                "review_id": review_id,
                "title": title,
                "body": body,
                "rating": rating,
                "author": author,
                "date": date,
            }
        )
    return reviews


def normalize_reviews(raw_reviews: List[Dict[str, Any]], product_meta: Dict[str, Any], max_reviews: int):
    rows: List[Dict[str, Any]] = []
    for idx, rev in enumerate(raw_reviews[:max_reviews]):
        rating = None
        if isinstance(rev.get("reviewRating"), dict):
            rating = rev["reviewRating"].get("ratingValue")
        if rating is None and "rating" in rev:
            rating = rev.get("rating")
        author = rev.get("author")
        if isinstance(author, dict):
            author = author.get("name")
        rows.append(
            {
                "product_id": product_meta.get("product_id"),
                "brand": product_meta.get("brand"),
                "product_name": product_meta.get("product_name"),
                "flavor": product_meta.get("flavor"),
                "size_variant": product_meta.get("size_variant"),
                "source_site": product_meta.get("source_site"),
                "source_url": product_meta.get("source_url"),
                "review_id": rev.get("@id") or rev.get("url") or rev.get("review_id") or f"{product_meta.get('product_id')}_{idx}",
                "title": rev.get("name") or rev.get("headline") or rev.get("title"),
                "body": rev.get("reviewBody") or rev.get("description") or rev.get("body"),
                "rating": rating,
                "author": author,
                "date": rev.get("datePublished") or rev.get("date"),
            }
        )
    return rows


def save_reviews(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[ok] Wrote {len(df)} reviews -> {out_path}")


def main():
    # Prefer Amazon scraping when available; otherwise parse saved HTML/JSON-LD.
    args = parse_args()
    products_df = load_products(Path(args.products))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_dir = Path(args.html_dir) if args.html_dir else None
    save_html_dir = Path(args.save_html_dir) if args.save_html_dir else None
    if save_html_dir:
        save_html_dir.mkdir(parents=True, exist_ok=True)

    for _, product in products_df.iterrows():
        meta = product.to_dict()
        product_id = meta.get("product_id")
        parsed_reviews: List[Dict[str, Any]] = []

        # Prefer Amazon scraping if the source is Amazon or user opts to force Amazon
        is_amazon = "amazon.com" in str(meta.get("source_url", "")).lower() or str(meta.get("source_site", "")).lower() == "amazon"
        reviews_url = None
        asin = None
        if is_amazon or args.force_amazon:
            res = build_amazon_reviews_url(meta, timeout=args.timeout, allow_search=args.amazon_search or args.force_amazon)
            if res:
                reviews_url, asin = res
                print(f"[info] Using Amazon reviews for {product_id} (ASIN={asin})")

        if reviews_url and args.amazon_pages > 0:
            for page in range(1, args.amazon_pages + 1):
                page_url = f"{reviews_url}&pageNumber={page}"
                html_page = fetch_html(page_url, timeout=args.timeout)
                if html_page and save_html_dir:
                    (save_html_dir / f"{product_id}_p{page}.html").write_text(html_page, encoding="utf-8")
                if not html_page:
                    print(f"[warn] Failed to fetch Amazon page {page} for {product_id}")
                    break
                page_reviews = parse_amazon_reviews(html_page)
                if not page_reviews:
                    print(f"[warn] No reviews parsed on Amazon page {page} for {product_id}; stopping pagination.")
                    break
                parsed_reviews.extend(page_reviews)
                if len(parsed_reviews) >= args.max_reviews:
                    break
                time.sleep(args.sleep)

        # If nothing from Amazon, try JSON-LD on provided/saved HTML
        if not parsed_reviews:
            html: Optional[str] = None
            if html_dir:
                # Allow multiple saved pages per product (e.g., product_id_p1.html, product_id_p2.html)
                candidates = sorted(html_dir.glob(f"{product_id}*.html"))
                if not candidates:
                    print(f"[warn] HTML file not found for {product_id} in {html_dir}")
                for cand in candidates:
                    html = cand.read_text(encoding="utf-8", errors="ignore")
                    print(f"[info] Loaded HTML for {product_id} from {cand}")
                    # Try Amazon parser first (handles saved Amazon review pages)
                    page_reviews = parse_amazon_reviews(html)
                    if not page_reviews:
                        page_reviews = extract_jsonld_reviews(html)
                    parsed_reviews.extend(page_reviews)
            if not parsed_reviews:
                html = fetch_html(meta["source_url"], timeout=args.timeout)
                if html and save_html_dir:
                    (save_html_dir / f"{product_id}.html").write_text(html, encoding="utf-8")
                    print(f"[info] Saved HTML for {product_id} to {save_html_dir / f'{product_id}.html'}")
                if html:
                    time.sleep(args.sleep)
                    parsed_reviews.extend(extract_jsonld_reviews(html))

        if not parsed_reviews:
            print(f"[warn] No reviews found for {product_id}. Consider saving HTML manually or adjusting source URLs.")
            continue

        rows = normalize_reviews(parsed_reviews, meta, max_reviews=args.max_reviews)
        out_path = out_dir / f"reviews_{product_id}.csv"
        save_reviews(rows, out_path)


if __name__ == "__main__":
    main()
