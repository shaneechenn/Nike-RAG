"""
PHASE 1 — Load the Nike CSV into SQLite and build the keyword-search index.

Works with the rich "Nike Products: Prices, Descriptions, Reviews" dataset
(name, sub_title, brand, color, price, avg_rating, review_count, raw_description, ...)
and also falls back to the simple 3-column dataset (Title, Subtitle, Product Description).

Run once:
    python load_data.py
"""

import glob
import html
import os
import re
import sqlite3

import pandas as pd

DATA_DIR = "data"
DB_PATH = "nike.db"


def find_csv():
    candidates = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    candidates += glob.glob("*.csv")
    if not candidates:
        raise SystemExit("No CSV found. Put the Nike CSV in ./data/ (or this folder).")
    return candidates[0]


def strip_html(text):
    """Remove HTML tags and decode entities (raw_description is often HTML)."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def to_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = re.sub(r"[^0-9.\-]", "", str(v))
    try:
        return float(s)
    except ValueError:
        return None


def detect_gender(text):
    """Extract a gender signal from text like "Men's Shoes"."""
    t = (text or "").lower()
    if "women" in t or "woman" in t or "girl" in t or "ladies" in t:
        return "women"
    if "men" in t or "boy" in t or "man" in t:
        return "men"
    if "kid" in t or "toddler" in t or "baby" in t or "child" in t or "infant" in t or "youth" in t:
        return "kids"
    return None


def main():
    csv_path = find_csv()
    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")

    lower = {str(c).lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None

    # Flexible mapping: rich dataset first, then the simple one.
    title_col = pick("name", "product name", "title", "product title", "product_title")
    subtitle_col = pick("sub_title", "subtitle")
    desc_col = pick("description", "raw_description", "product description")
    brand_col = pick("brand")
    color_col = pick("color")
    model_col = pick("model", "sku", "product id")
    price_col = pick("price", "sale price", "listing price")
    rating_col = pick("avg_rating", "rating")
    reviews_col = pick("review_count", "reviews", "review count")

    if title_col is None:
        title_col = df.columns[0]
    if desc_col is None and len(df.columns) >= 3:
        desc_col = df.columns[2]

    print(f"Using -> title: {title_col!r}, subtitle: {subtitle_col!r}, "
          f"description: {desc_col!r}, price: {price_col!r}, rating: {rating_col!r}")

    def get(row, col):
        if col is None:
            return None
        v = row.get(col)
        return None if pd.isna(v) else v

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS products")
    conn.execute("DROP TABLE IF EXISTS products_fts")
    conn.execute(
        """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT,
            description TEXT,
            brand TEXT,
            color TEXT,
            price REAL,
            avg_rating REAL,
            review_count INTEGER,
            gender TEXT,
            combined_text TEXT
        )
        """
    )
    conn.execute(
        "CREATE VIRTUAL TABLE products_fts USING fts5(title, subtitle, description)"
    )

    n = 0
    for _, r in df.iterrows():
        title = str(get(r, title_col) or "").strip()
        subtitle = str(get(r, subtitle_col) or "").strip()
        desc = strip_html(get(r, desc_col))
        brand = str(get(r, brand_col) or "").strip()
        color = str(get(r, color_col) or "").strip()
        model = str(get(r, model_col) or "").strip()
        price = to_float(get(r, price_col))
        rating = to_float(get(r, rating_col))
        reviews_raw = get(r, reviews_col)
        try:
            reviews = int(float(reviews_raw)) if reviews_raw is not None else None
        except (ValueError, TypeError):
            reviews = None

        gender = detect_gender(subtitle) or detect_gender(title)

        # Weight title/subtitle more (they carry the strongest "what is this /
        # who it's for" signal) and truncate the long marketing description.
        short_desc = (desc or "")[:220]
        parts = [title, title, subtitle, subtitle, brand, color, model, short_desc]
        combined = " . ".join(x for x in parts if x)

        cur = conn.execute(
            "INSERT INTO products (title, subtitle, description, brand, color, "
            "price, avg_rating, review_count, gender, combined_text) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (title, subtitle, desc, brand, color, price, rating, reviews, gender, combined),
        )
        conn.execute(
            "INSERT INTO products_fts (rowid, title, subtitle, description) "
            "VALUES (?,?,?,?)",
            (cur.lastrowid, title, subtitle, desc),
        )
        n += 1

    conn.commit()
    print(f"Done. {n} products stored in {DB_PATH}.")
    conn.close()


if __name__ == "__main__":
    main()
