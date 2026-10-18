"""
Fetch Nike products from Nike's (unofficial) product-feed API and save to CSV.

Uses the same JSON endpoint that nike.com uses. Each product includes an
explicit gender label (MEN / WOMEN / BOYS / GIRLS / ...), price, color, type.

Run:
    python fetch_nike.py
"""

import csv
import time

import requests

CHANNEL = "d9a5bc42-4b9c-4976-858a-f159cf99c647"
BASE = "https://api.nike.com/product_feed/rollup_threads/v2/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Broad searches to sample many categories (men's / women's / kids).
SEARCHES = [
    "shoes", "running shoes", "basketball shoes", "sneakers", "boots", "sandals",
    "t-shirt", "hoodie", "jacket", "pants", "shorts", "socks", "hat", "bag",
    "mens shoes", "womens shoes", "kids shoes", "kids clothes",
]
PER_SEARCH = 150  # max products to grab per search term
PAGE = 50
DELAY = 0.3  # polite pause between requests


def fetch(search, anchor, count):
    params = {
        "filter": ["marketplace(US)", "language(en)", "employeePrice(true)"],
        "searchTerms": search,
        "anchor": anchor,
        "count": count,
        "consumerChannelId": CHANNEL,
    }
    r = requests.get(BASE, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def extract(data, seen, rows):
    for obj in data.get("objects", []):
        for pi in obj.get("productInfo", []):
            mp = pi.get("merchProduct", {})
            pc = pi.get("productContent", {})
            price = pi.get("merchPrice", {})
            slug = pc.get("slug") or obj.get("seo", {}).get("slug")
            if not slug or slug in seen:
                continue
            seen.add(slug)
            rows.append(
                {
                    "title": pc.get("title") or obj.get("title") or "",
                    "subtitle": pc.get("subtitle") or obj.get("subtitle") or "",
                    "gender": "|".join(mp.get("genders", [])),
                    "product_type": mp.get("productType") or "",
                    "price": price.get("currentPrice"),
                    "currency": price.get("currency") or "USD",
                    "color": pc.get("colorDescription") or "",
                    "brand": mp.get("brand") or "Nike",
                    "sport": "|".join(mp.get("sportTags", [])),
                    "url": "https://www.nike.com/t/" + (slug or ""),
                }
            )


def main():
    seen = set()
    rows = []
    for s in SEARCHES:
        got = 0
        for anchor in range(0, PER_SEARCH, PAGE):
            try:
                data = fetch(s, anchor, PAGE)
                before = len(rows)
                extract(data, seen, rows)
                got += len(rows) - before
                print(f"  '{s}' anchor={anchor}: +{len(rows) - before} (total {len(rows)})")
                if len(data.get("objects", [])) < PAGE:
                    break
                time.sleep(DELAY)
            except Exception as e:  # noqa: BLE001
                print(f"  '{s}' anchor={anchor} failed: {e}")
                break
        time.sleep(DELAY)

    rows.sort(key=lambda r: r["title"])
    path = "data/nike_catalog.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} products to {path}")


if __name__ == "__main__":
    main()
