# CatFoodReview
Lightweight pipeline to fetch, clean, analyze, and summarize dry cat food reviews with sentiment and keyword/theme extraction.

## Project Structure
- `data/raw/`: source data (products.csv, downloaded datasets/html, raw reviews)
- `data/clean/`: cleaned and sentiment-labeled reviews
- `data/outputs/`: keywords, comparisons, figures
- `src/`: scripts (fetch, clean, sentiment, keywords, aggregate, visualize)

## Setup
```bash
pip install -r requirements.txt
```

## Data Fetch (open dataset)
Downloads UCSD Amazon Pet Supplies 5-core dataset, filters by ASINs, and writes per-product raw reviews.
```bash
python src/fetch_dataset.py --products data/raw/products.csv --out_dir data/raw/ --max_per_product 400 --max_scan 3000000 --review_urls data/raw/review_urls.csv
```

To obtain the open dataset manually (if needed):
- Download `https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Pet_Supplies_5.json.gz` into `data/raw/`.
- Optionally download metadata `https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles/meta_Pet_Supplies.json.gz` into `data/raw/`.

Exploring more brands/products:
- Add rows to `data/raw/products.csv` (product_id, brand, product_name, flavor, size, notes).
- Add matching ASINs to `data/raw/review_urls.csv` (`product_id,asin,...`). You can find ASINs via the metadata file or by searching Amazon product pages.
- Rerun the pipeline:
  ```
  python src/fetch_dataset.py --products data/raw/products.csv --review_urls data/raw/review_urls.csv --out_dir data/raw/ --max_per_product 400 --max_scan 3000000
  python src/clean.py --inp data/raw --out data/clean --min_length 20
  python src/sentiment.py --inp data/clean --out data/clean
  python src/keywords.py --inp data/clean --out data/outputs --products data/raw/products.csv --min_df 1 --top_k 10
  python src/aggregate.py --clean_dir data/clean --keywords data/outputs/keywords.csv --products data/raw/products.csv --out data/outputs
  python src/visualize.py --comparison data/outputs/comparison.csv --keywords data/outputs/keywords.csv --out_dir data/outputs/figures --top_n 10
  ```

## Cleaning
Normalize text, drop short/duplicate reviews, coerce ratings, parse dates.
```bash
python src/clean.py --inp data/raw --out data/clean --min_length 20
```

## Sentiment
Apply VADER to cleaned reviews.
```bash
python src/sentiment.py --inp data/clean --out data/clean
```

## Keywords
Extract TF-IDF keywords per product and sentiment bucket (with extra stopwords and brand filtering).
```bash
python src/keywords.py --inp data/clean --out data/outputs --products data/raw/products.csv --min_df 1 --top_k 10
```

## Aggregation
Compute sentiment metrics, rating breakdowns, simple score (pos% - neg%), and attach keywords.
```bash
python src/aggregate.py --clean_dir data/clean --keywords data/outputs/keywords.csv --products data/raw/products.csv --out data/outputs
```

## Visualization
Generate PNGs for sentiment distribution, rating score/lines, rating distribution, and top terms per product.
```bash
python src/visualize.py --comparison data/outputs/comparison.csv --keywords data/outputs/keywords.csv --out_dir data/outputs/figures --top_n 10
```

