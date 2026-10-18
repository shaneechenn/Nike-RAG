"""
PHASE 1 — Load the Nike CSV into SQLite and build the keyword-search index.

Works with the Nike API catalog (title, subtitle, gender, product_type, price,
color, brand, sport) and older datasets (Title/Subtitle/Product Description).

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

    title_col = pick("name", "product name", "title", "product title", "product_title")
    subtitle_col = pick("sub_title", "subtitle")
    desc_col = pick("description", "raw_description", "product description")
    brand_col = pick("brand")
    color_col = pick("color")
    gender_col = pick("gender")
    type_col = pick("product_type", "product type")
    sport_col = pick("sport")
    url_col = pick("url")
    model_col = pick("model", "sku", "product id")
    price_col = pick("price", "sale price", "listing price")

    if title_col is None:
        title_col = df.columns[0]

    print(f"Using -> title: {title_col!r}, subtitle: {subtitle_col!r}, "
          f"gender: {gender_col!r}, price: {price_col!r}")

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
            gender TEXT,
            product_type TEXT,
            sport TEXT,
            url TEXT,
            price REAL,
            combined_text TEXT
        )
        """
    )
    conn.execute(
        "CREATE VIRTUAL TABLE products_fts USING fts5(title, subtitle, description, color)"
    )

    n = 0
    for _, r in df.iterrows():
        title = str(get(r, title_col) or "").strip()
        subtitle = str(get(r, subtitle_col) or "").strip()
        desc = strip_html(get(r, desc_col))
        brand = str(get(r, brand_col) or "").strip()
        color = str(get(r, color_col) or "").strip()
        gender = str(get(r, gender_col) or "").strip()
        ptype = str(get(r, type_col) or "").strip()
        sport = str(get(r, sport_col) or "").strip()
        model = str(get(r, model_col) or "").strip()
        price = to_float(get(r, price_col))

        # Weight title/subtitle more; include gender/type/sport for semantics.
        gender_clean = gender.replace("|", " ")
        url = str(get(r, url_col) or "").strip()
        short_desc = (desc or "")[:200]
        parts = [title, title, subtitle, subtitle, gender_clean, ptype, sport,
                 brand, color, model, short_desc]
        combined = " . ".join(x for x in parts if x)

        cur = conn.execute(
            "INSERT INTO products (title, subtitle, description, brand, color, gender, "
            "product_type, sport, url, price, combined_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (title, subtitle, desc, brand, color, gender, ptype, sport, url,
             price, combined),
        )
        conn.execute(
            "INSERT INTO products_fts (rowid, title, subtitle, description, color) "
            "VALUES (?,?,?,?,?)",
            (cur.lastrowid, title, subtitle, desc, color),
        )
        n += 1

    conn.commit()
    print(f"Done. {n} products stored in {DB_PATH}.")
    conn.close()


if __name__ == "__main__":
    main()
