# ResearchMind — Python + Google Gemini edition

AI research assistant for academics, rewritten as a **100% Python** app (no Node.js, no Rust/Tauri, no React).
All AI runs on **Google Gemini** models through your Google API key.

## Features
| Page | What it does |
|---|---|
| 📥 Import | Drag & drop PDF, DOCX, EPUB, TXT, MD, HTML → parse → OCR scanned pages (Gemini vision) → chunk → embed (Gemini embeddings) → index → AI summary, title/authors/year/DOI/tags. Retry failed files. |
| 📚 Library | Collections, tags, stars, read status, edit metadata, AI summaries, abstracts, daily reading suggestion. |
| 🤖 AI Chat | Streaming RAG chat over one, several or all papers, with `[Title, page X]` citations and sources. Modes: Q&A, **Critique**, **Debate (Pro vs Con)**, **Compare papers**, General. Saved conversations. |
| 🔍 Search | Hybrid search: BM25 (SQLite FTS5) + Gemini vector similarity, fused with RRF. Save results as highlights. |
| 📝 Review Builder | 7-section literature review (Background → Future Directions) + bibliography (APA/IEEE). Edit inline, regenerate one section, save, export **DOCX / HTML / Markdown**. |
| ✅ Verify & Discover | Verify DOIs/references via OpenAlex, Crossref, Semantic Scholar (retraction check); check a claim against your papers; discover related papers. |
| 🖍️ Notes | Highlights & notes per paper, searchable, exportable. |
| ⚙️ Settings | API key, pick chat / fallback / embedding models (loads the live list from your key), temperature, answer language, and more. |

## Quick start

**Requirements:** Python 3.10+ and a Google API key from https://aistudio.google.com/apikey

### Windows
Double-click `run.bat` (first run creates a virtual env and installs everything).

### macOS / Linux
```bash
./run.sh
```

### Manual
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env        # (cp on macOS/Linux) then put your key in .env
streamlit run app.py
```
The app opens at http://localhost:8501. You can also paste the API key on the **⚙️ Settings** page instead of using `.env`.

## Models (defaults, changeable in Settings or `.env`)
- Chat: `gemini-3.8-flash` (fallback `gemini-3.5-flash-lite`; `gemini-3.1-pro-preview` for highest quality)
- Embeddings: `gemini-embedding-001` (768 dims)

If you change the embedding model or dimensions, click **Re-index whole library** in Settings.

## Project layout
```
app.py                  Streamlit UI (all pages)
researchmind/
  config.py             settings (.env + data/settings.json)
  gemini.py             Google Gen AI SDK wrapper: generate, stream, embed, OCR, model list, fallback/retry
  parser.py             PDF/DOCX/EPUB/TXT/MD/HTML parsing + metadata guessing
  chunker.py            page-aware chunking with overlap
  ingest.py             import pipeline
  db.py                 SQLite storage (papers, chunks+vectors, FTS5, notes, chats, reviews)
  search.py             hybrid BM25 + vector search (RRF)
  rag.py                chat / critique / debate / compare / review prompts
  verify.py             OpenAlex, Crossref, Semantic Scholar, claim verification
  export.py             Markdown / HTML / DOCX export
data/                   created automatically: database, uploaded papers, settings
```

## Privacy
Your files and database stay in the local `data/` folder. Text excerpts are sent to the Google Gemini API
for embeddings and answers, and DOI lookups go to OpenAlex/Crossref/Semantic Scholar.
