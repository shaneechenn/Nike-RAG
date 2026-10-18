# 👟 Nike Product RAG Assistant

A chat app that recommends Nike products using **hybrid search**:

- **Keyword search** (SQLite FTS5 + BM25) — matches product words like `shoes`, `red`, or `basketball`.
- **Semantic search** (sentence-transformers `all-mpnet-base-v2` + Chroma) — understands meaning like `gift for dad`.
- **Cross-encoder re-ranker** (`ms-marco-MiniLM-L-6-v2`) — re-scores the top candidates for precise relevance.
- **Recipient-aware routing** — detects who the gift is for ("dad" → men's, "aunt" → women's, "son" → kids') by projecting the query onto the embedding model's gender/age axes. No hard-coded word lists — it generalizes to any word the model knows.
- **Multi-intent queries** — `basketball shoes or volleyball shoes` returns *both*, `red or blue shoes` returns *both* colors.

## Tech stack

| Piece | Tool |
|---|---|
| Frontend | HTML + CSS + JavaScript (`index.html`) |
| Backend | FastAPI (`server.py`) |
| Keyword search | SQLite FTS5 (BM25, OR-based) |
| Embeddings | sentence-transformers `all-mpnet-base-v2` (768-dim) |
| Re-ranker | sentence-transformers `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector DB | Chroma |
| Environment | conda env `nike-rag` (Python 3.11) |
| Product data | Nike's product-feed API (fetched by `fetch_nike.py`) |

## Run it (easiest)

Double-click **`run.bat`**, then open **http://localhost:8501** in your browser.
Keep the black window open; press `Ctrl+C` in it to stop.

Or manually (from the `nike-rag` folder):

```powershell
<path>\nike-rag\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8501
```

## Project layout

```
nike-rag/
  index.html        # the chat UI (HTML/CSS/JS)
  server.py         # FastAPI backend (serves the page + /search)
  search.py         # keyword / semantic / cross-encoder / recipient routing
  load_data.py      # CSV -> SQLite + FTS5 keyword index
  embed.py          # text -> vectors -> Chroma
  fetch_nike.py     # pulls the product catalog from Nike's API
  make_avatars.py   # generates the chat avatars (optional)
  user_avatar.png   # chat avatars (used by index.html)
  assistant_avatar.png
  run.bat           # double-click to launch
  setup.ps1         # creates the conda env + installs deps
  requirements.txt
  render.yaml       # Render.com deploy config
  data/nike_catalog.csv  # the catalog (1,891 products, fetched from Nike's API)
  nike.db           # built by load_data.py (regular DB)
  chroma_db/        # built by embed.py (vector DB)
```

## Setup from scratch (only if moving to a new machine)

### 1. Install Miniconda

```powershell
& "$env:USERPROFILE\OneDrive\Documents\deepseek-harness\default-workspace\Miniconda3-latest-Windows-x86_64.exe" /S /InstallationType=JustMe /RegisterPython=0 /AddToPath=1 /D=C:\Users\shane\miniconda3
```

Then close and reopen PowerShell.

### 2. Create the environment + install dependencies

From the `nike-rag` folder:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

### 3. Fetch the catalog + build the search index

```powershell
<path>\nike-rag\python.exe fetch_nike.py   # pulls products from Nike's API
<path>\nike-rag\python.exe load_data.py    # CSV -> SQLite + FTS5
<path>\nike-rag\python.exe embed.py        # text -> vectors -> Chroma
```

> First run downloads two models: `all-mpnet-base-v2` (~420 MB) and the
> cross-encoder (~80 MB). They are cached afterwards.

## How the search works

For a query like `shoes for dad`:

1. **Keyword** — extracts content words (drops fillers like "for", "a", "gift"),
   matches them against titles, subtitles, descriptions **and colors** with
   SQLite FTS5 (OR + BM25) — so `red or blue shoes` matches real colorways.
2. **Semantic re-rank** — re-orders the keyword candidates by cosine similarity
   with the query embedding (`all-mpnet-base-v2`).
3. **Recipient detection** — projects each query word onto the model's
   gender/age axes ("dad" → male+adult, "aunt" → female+adult, "son" → child).
   If a recipient is detected, matching products (by Nike's own MEN/WOMEN/KIDS
   label) are pulled in and ranked first.
4. **Cross-encoder** — re-scores the top ~20 candidates with
   `ms-marco-MiniLM-L-6-v2` for final relevance ordering.

Queries containing **or / and** (`basketball shoes or volleyball shoes`) are
split into separate searches and the results interleaved, so every option
appears at least once.

## Publish it for free (get a public link)

The easiest free host is **Hugging Face Spaces** (2 CPU / 16 GB RAM — enough for
the models). You get a public link like
`https://huggingface.co/spaces/<your-username>/nike-rag`.

1. Create a free account at <https://huggingface.co/join>.
2. Create a Space at <https://huggingface.co/new-space>: name `nike-rag`,
   **SDK: Docker**, **template: Blank**, visibility **Public**.
3. Push this folder to the Space (run from the `nike-rag` folder):

   ```powershell
   git remote add space https://huggingface.co/spaces/<YOUR-USERNAME>/nike-rag
   git push space master:main
   ```

   When asked for a password, use an access token from
   <https://huggingface.co/settings/tokens> (role: **Write**).
4. Wait for the build (~10–15 minutes the first time — it installs everything,
   downloads the two AI models, and builds the search index).
5. Open your link and share it. The first visit after the Space has been idle
   may take ~30 seconds to wake up.

The `render.yaml` is kept as an alternative for **Render.com** (note: Render's
free tier has only 512 MB RAM, which is not enough for these models).
