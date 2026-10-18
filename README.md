# 👟 Nike Product RAG Assistant

A small chatbot that recommends Nike products using **hybrid search**:

- **Keyword search** (SQLite FTS5 / BM25) — finds exact product names like "Nike Air Force 1".
- **Semantic search** (sentence-transformers + Chroma) — finds meaning, like "a good gift for my mom".
- **Hybrid** = both, merged with Reciprocal Rank Fusion.
- Optional **LLM** answer (OpenAI) that writes a friendly recommendation from the top matches.

## The tech, in one line each

| Piece | Tool | What it does |
|---|---|---|
| Regular DB + keyword search | SQLite (built into Python) | Stores products + exact-word search |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Turns text into meaning-vectors |
| Vector DB | Chroma | Finds the closest-meaning products |
| Chat UI | Streamlit | The chat box you type into |
| AI answer (optional) | OpenAI `gpt-4o-mini` | Writes the friendly recommendation |

## Setup (one time)

### 1. Install Miniconda

If not already installed, run this in PowerShell:

```powershell
& "$env:USERPROFILE\OneDrive\Documents\deepseek-harness\default-workspace\Miniconda3-latest-Windows-x86_64.exe" /S /InstallationType=JustMe /RegisterPython=0 /AddToPath=1 /D=C:\Users\shane\miniconda3
```

Then close and reopen PowerShell so `conda` is on your PATH.

### 2. Create the environment and install dependencies

From this `nike-rag` folder:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

### 3. Put the dataset CSV here

The CSV already lives in `data/nike_data_2022_09.csv` — the **"Nike Products:
Prices, Descriptions, Reviews"** dataset (Crawl Feeds / Kaggle). `load_data.py`
auto-detects columns, so it also works with the simpler 3-column dataset if you
swap it in.

> Columns: `name, sub_title, brand, model, color, price, currency, availability,
> description, avg_rating, review_count, images, available_sizes` (112 products).

## Running

```powershell
conda activate nike-rag

python load_data.py     # 1) CSV -> SQLite + keyword index
python embed.py         # 2) build embeddings -> Chroma (downloads ~90MB model on first run)
streamlit run app.py    # 3) open the chat UI
```

Then type either:
- `I want the Nike Air Force 1s`
- `what's a good gift for my mom?`

### Optional: turn on the AI-written answer

```powershell
$env:OPENAI_API_KEY = "sk-..."
streamlit run app.py
```

Without a key, the app still works — it just shows the ranked product list
without a written recommendation.

## Project layout

```
nike-rag/
  load_data.py   # CSV -> SQLite (products + FTS5 keyword index)
  embed.py       # text -> embeddings -> Chroma vector DB
  search.py      # keyword / semantic / hybrid / LLM answer
  app.py         # Streamlit chat UI
  setup.ps1      # creates the conda env + installs deps
  requirements.txt
  nike.db        # created by load_data.py (regular DB)
  chroma_db/     # created by embed.py (vector DB)
```
