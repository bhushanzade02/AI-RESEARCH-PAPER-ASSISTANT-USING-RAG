"""ResearchMind (Python edition) — AI research assistant powered by Google Gemini.

Run:  streamlit run app.py
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from researchmind import db
from researchmind.config import DEFAULT_CHAT_MODELS, DEFAULT_EMBED_MODELS, Settings
from researchmind.export import markdown_to_docx, markdown_to_html, review_to_markdown
from researchmind.gemini import Gemini, GeminiError
from researchmind.ingest import ingest, save_upload
from researchmind.parser import SUPPORTED
from researchmind.rag import REVIEW_SECTIONS, bibliography, chat_stream, generate_review_section
from researchmind.search import hybrid_search
from researchmind.verify import search_literature, verify_claim, verify_reference

st.set_page_config(page_title="ResearchMind", page_icon="📚", layout="wide")
db.init_db()

if "settings" not in st.session_state:
    st.session_state.settings = Settings.load()
S: Settings = st.session_state.settings


def get_gem(required: bool = True) -> Gemini | None:
    try:
        return Gemini(S)
    except GeminiError as e:
        if required:
            st.error(f"{e}")
            st.stop()
        return None


def paper_label(p: dict) -> str:
    return f"{p['title'][:90]} ({p.get('year') or 'n.d.'})"


def paper_picker(key: str, label: str = "Papers (leave empty = whole library)", default_all=False) -> list[str]:
    papers = [p for p in db.list_papers() if p["status"] == "ready"]
    cols = db.list_collections()
    c1, c2 = st.columns([1, 3])
    coll = c1.selectbox("Collection", ["(all)"] + cols, key=f"{key}_coll")
    if coll != "(all)":
        papers = [p for p in papers if p["collection"] == coll]
    opts = {p["id"]: paper_label(p) for p in papers}
    sel = c2.multiselect(label, list(opts), format_func=lambda x: opts[x], key=f"{key}_papers",
                         default=list(opts) if default_all else None)
    if not sel and coll != "(all)":
        return list(opts)
    return sel


def show_sources(hits: list[dict]) -> None:
    if not hits:
        return
    with st.expander(f"📎 Sources ({len(hits)} excerpts)"):
        for i, h in enumerate(hits, 1):
            st.markdown(f"**[{i}] {h['title']}** — page {h.get('page') or '?'}"
                        + (f" · _{h['section']}_" if h.get("section") else ""))
            st.caption(h["content"][:600] + ("…" if len(h["content"]) > 600 else ""))


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📚 ResearchMind")
st.sidebar.caption("Python + Google Gemini edition")
PAGE = st.sidebar.radio("Go to", ["📥 Import", "📚 Library", "🤖 AI Chat", "🔍 Search", "📝 Review Builder",
                                  "✅ Verify & Discover", "🖍️ Notes & Highlights", "⚙️ Settings"])
stt = db.stats()
st.sidebar.markdown(f"**{stt['papers']}** papers · **{stt['chunks']}** chunks · **{stt['notes']}** notes")
if not S.google_api_key:
    st.sidebar.warning("No Google API key — open ⚙️ Settings.")
else:
    st.sidebar.success(f"Model: {S.chat_model}")


# ================================================================= IMPORT
if PAGE == "📥 Import":
    st.header("📥 Smart Import")
    st.write("Drag & drop PDF, DOCX, EPUB, TXT, MD or HTML. Files are parsed, chunked, embedded with Gemini, "
             "indexed and summarized.")
    files = st.file_uploader("Files", type=[e.strip(".") for e in SUPPORTED], accept_multiple_files=True)
    c1, c2, c3 = st.columns(3)
    collection = c1.text_input("Add to collection (optional)", "")
    use_ocr = c2.checkbox("OCR scanned pages with Gemini vision", True)
    do_sum = c3.checkbox("Auto-summarize + extract metadata", True)
    if st.button("Import", type="primary", disabled=not files):
        gem = get_gem()
        for f in files:
            with st.status(f"Importing {f.name}", expanded=True) as status:
                path = save_upload(f.name, f.getvalue())
                pid = ingest(path, S, gem, collection=collection.strip(), use_ocr=use_ocr,
                             summarize=do_sum, log=st.write)
                p = db.get_paper(pid)
                status.update(label=f"{f.name}: {p['status']}",
                              state="complete" if p["status"] == "ready" else "error")
        st.success("Import finished. Open 📚 Library or 🤖 AI Chat.")

    st.subheader("Import queue / status")
    for p in db.list_papers():
        if p["status"] != "ready":
            c1, c2, c3 = st.columns([5, 2, 1])
            c1.write(f"**{p['title']}** — `{p['file_name']}`")
            c2.write(f"Status: `{p['status']}` {p['error'][:120]}")
            if c3.button("Retry", key=f"retry_{p['id']}"):
                with st.spinner("Retrying…"):
                    ingest(Path(p["file_path"]), S, get_gem(), pid=p["id"], log=lambda *_: None)
                st.rerun()

# ================================================================ LIBRARY
elif PAGE == "📚 Library":
    st.header("📚 Library & Collections")
    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    q = c1.text_input("Filter by title / author / tag")
    coll = c2.selectbox("Collection", ["(all)"] + db.list_collections())
    starred = c3.checkbox("⭐ only")
    rs = c4.selectbox("Status", ["(any)", "unread", "reading", "read"])
    papers = db.list_papers(None if coll == "(all)" else coll, q, starred, None if rs == "(any)" else rs)
    st.caption(f"{len(papers)} papers")
    for p in papers:
        star = "⭐ " if p["starred"] else ""
        with st.expander(f"{star}{paper_label(p)} · {p['n_chunks']} chunks · {p['status']}"):
            if p["authors"]:
                st.write(f"**Authors:** {p['authors']}")
            if p["doi"]:
                st.write(f"**DOI:** [{p['doi']}](https://doi.org/{p['doi']})")
            if p["tags"]:
                st.write(f"**Tags:** {p['tags']}")
            if p["summary"]:
                st.markdown("**AI summary**\n\n" + p["summary"])
            if p["abstract"]:
                with st.popover("Abstract"):
                    st.write(p["abstract"])
            with st.form(f"edit_{p['id']}"):
                a, b = st.columns(2)
                title = a.text_input("Title", p["title"])
                authors = b.text_input("Authors", p["authors"])
                a, b, c, d = st.columns(4)
                year = a.number_input("Year", 0, 2100, int(p["year"] or 0))
                doi = b.text_input("DOI", p["doi"])
                coll_in = c.text_input("Collection", p["collection"])
                status = d.selectbox("Read status", ["unread", "reading", "read"],
                                     index=["unread", "reading", "read"].index(p["read_status"] or "unread"))
                tags = st.text_input("Tags (comma separated)", p["tags"])
                s1, s2, s3 = st.columns(3)
                if s1.form_submit_button("💾 Save"):
                    db.update_paper(p["id"], title=title, authors=authors, year=year or None, doi=doi,
                                    collection=coll_in, read_status=status, tags=tags)
                    st.rerun()
                if s2.form_submit_button("⭐ Toggle star"):
                    db.update_paper(p["id"], starred=0 if p["starred"] else 1)
                    st.rerun()
                if s3.form_submit_button("🗑️ Delete"):
                    db.delete_paper(p["id"])
                    st.rerun()
            fp = Path(p["file_path"])
            if fp.exists():
                st.download_button("⬇️ Download original", fp.read_bytes(), file_name=p["file_name"],
                                   key=f"dl_{p['id']}")

    st.divider()
    st.subheader("✨ Daily reading suggestion")
    unread = [p for p in db.list_papers(read_status="unread") if p["status"] == "ready"]
    if unread:
        p = unread[hash(str(__import__("datetime").date.today())) % len(unread)]
        st.info(f"Today, read: **{paper_label(p)}**\n\n{p['summary'][:600]}")
    else:
        st.caption("No unread papers.")

# ================================================================== CHAT
elif PAGE == "🤖 AI Chat":
    st.header("🤖 AI Chat & RAG")
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = "chat-" + uuid.uuid4().hex[:6]
    top = st.columns([2, 2, 1, 1])
    mode_label = top[0].selectbox("Mode", ["Q&A with papers", "Critique", "Debate (Pro vs Con)",
                                           "Compare papers", "General (no documents)"])
    mode = {"Q&A with papers": "qa", "Critique": "critique", "Debate (Pro vs Con)": "debate",
            "Compare papers": "compare", "General (no documents)": "general"}[mode_label]
    sessions = db.list_chat_sessions()
    cur = st.session_state.chat_session
    choice = top[1].selectbox("Conversation", [cur] + [s for s in sessions if s != cur])
    if choice != cur:
        st.session_state.chat_session = choice
        st.rerun()
    if top[2].button("➕ New"):
        st.session_state.chat_session = "chat-" + uuid.uuid4().hex[:6]
        st.rerun()
    if top[3].button("🗑️ Clear"):
        db.clear_chat(st.session_state.chat_session)
        st.rerun()

    paper_ids = paper_picker("chat") if mode != "general" else []
    session = st.session_state.chat_session
    history = db.get_chat(session)
    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                show_sources(m["meta"].get("sources", []))

    placeholder = {"qa": "Ask a question about your papers…",
                   "critique": "Optional focus for the critique (or just press enter with 'critique')",
                   "debate": "Position to debate, e.g. 'Transformers beat CNNs for small datasets'",
                   "compare": "Optional focus for the comparison",
                   "general": "Ask anything…"}[mode]
    prompt = st.chat_input(placeholder)
    if prompt:
        gem = get_gem()
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                stream, hits = chat_stream(prompt, [{"role": m["role"], "content": m["content"]} for m in history],
                                           gem, S, paper_ids, mode)
                answer = st.write_stream(stream)
                show_sources(hits)
            except GeminiError as e:
                answer, hits = f"⚠️ {e}", []
                st.error(answer)
        db.add_chat(session, "user", prompt, {"mode": mode})
        db.add_chat(session, "assistant", answer if isinstance(answer, str) else str(answer),
                    {"mode": mode, "sources": hits, "model": getattr(gem, "last_model", S.chat_model)})
    if history:
        md = "\n\n".join(f"**{m['role'].title()}:** {m['content']}" for m in history)
        st.download_button("⬇️ Export conversation (.md)", md, file_name=f"{session}.md")

# ================================================================ SEARCH
elif PAGE == "🔍 Search":
    st.header("🔍 Hybrid Semantic Search")
    st.caption("BM25 full-text (SQLite FTS5) + Gemini embeddings, fused with Reciprocal Rank Fusion.")
    q = st.text_input("Search query")
    c1, c2 = st.columns([3, 1])
    with c1:
        pids = paper_picker("search")
    k = c2.slider("Results", 3, 30, 10)
    if q:
        gem = get_gem(required=False)
        hits = hybrid_search(q, gem, pids or None, top_k=k, max_per_paper=k)
        if not hits:
            st.info("No matches.")
        for h in hits:
            with st.container(border=True):
                st.markdown(f"**{h['title']}** — page {h.get('page') or '?'}"
                            + (f" · _{h['section']}_" if h.get("section") else "")
                            + (f" · similarity {h['similarity']:.2f}" if h.get("similarity") else ""))
                st.write(h["content"][:900])
                if st.button("🖍️ Save as highlight", key=f"hl_{h['id']}"):
                    db.add_note(h["paper_id"], h["content"][:1000], "", h.get("page"))
                    st.toast("Saved to Notes & Highlights")

# ======================================================== REVIEW BUILDER
elif PAGE == "📝 Review Builder":
    st.header("📝 Literature Review Builder")
    tab_new, tab_saved = st.tabs(["Build", "Saved reviews"])
    with tab_new:
        pids = paper_picker("review", "Select papers for the review")
        topic = st.text_input("Review topic / title", "Literature Review")
        extra = st.text_area("Extra instructions (optional)", placeholder="e.g. focus on clinical applications")
        secs = st.multiselect("Sections", list(REVIEW_SECTIONS), default=list(REVIEW_SECTIONS),
                              format_func=lambda k: REVIEW_SECTIONS[k]["title"])
        style = st.radio("Bibliography style", ["APA", "IEEE"], horizontal=True)
        if "review" not in st.session_state:
            st.session_state.review = {}
        if st.button("🚀 Generate review", type="primary", disabled=not pids):
            gem = get_gem()
            st.session_state.review = {}
            for k in list(st.session_state.keys()):
                if str(k).startswith("rev_"):
                    del st.session_state[k]
            prog = st.progress(0.0)
            for i, key in enumerate(secs):
                prog.progress(i / len(secs), text=f"Writing {REVIEW_SECTIONS[key]['title']}…")
                try:
                    text, _ = generate_review_section(key, pids, gem, S, topic, extra)
                except GeminiError as e:
                    text = f"⚠️ {e}"
                st.session_state.review[key] = {"title": REVIEW_SECTIONS[key]["title"], "content": text}
            st.session_state.review["bibliography"] = {"title": "8. Bibliography",
                                                       "content": bibliography(pids, style)}
            st.session_state.review_meta = {"topic": topic, "pids": pids}
            prog.progress(1.0, text="Done")

        rev = st.session_state.review
        if rev:
            meta = st.session_state.get("review_meta", {"topic": topic, "pids": pids})
            for key, sec in list(rev.items()):
                with st.expander(sec["title"], expanded=True):
                    new = st.text_area("Edit", sec["content"], height=260, key=f"rev_{key}",
                                       label_visibility="collapsed")
                    rev[key]["content"] = new
                    if key != "bibliography" and st.button("🔄 Regenerate", key=f"regen_{key}"):
                        with st.spinner("Regenerating…"):
                            text, _ = generate_review_section(key, meta["pids"], get_gem(), S, meta["topic"], extra)
                        rev[key]["content"] = text
                        st.session_state.pop(f"rev_{key}", None)
                        st.rerun()
            md = review_to_markdown(meta["topic"], rev)
            c1, c2, c3, c4 = st.columns(4)
            c1.download_button("⬇️ Markdown", md, file_name="review.md")
            c2.download_button("⬇️ HTML", markdown_to_html(md, meta["topic"]), file_name="review.html")
            c3.download_button("⬇️ DOCX", markdown_to_docx(md), file_name="review.docx")
            if c4.button("💾 Save review"):
                db.save_review(meta["topic"], meta["pids"], rev)
                st.success("Saved.")
    with tab_saved:
        for r in db.list_reviews():
            with st.expander(f"{r['title']} · {r['created_at']}"):
                md = review_to_markdown(r["title"], r["sections"])
                st.markdown(md)
                c1, c2, c3 = st.columns(3)
                c1.download_button("⬇️ DOCX", markdown_to_docx(md), file_name="review.docx", key=f"d_{r['id']}")
                if c2.button("✏️ Load into editor", key=f"l_{r['id']}"):
                    st.session_state.review = r["sections"]
                    for k in list(st.session_state.keys()):
                        if str(k).startswith("rev_"):
                            del st.session_state[k]
                    st.session_state.review_meta = {"topic": r["title"], "pids": r["paper_ids"]}
                    st.rerun()
                if c3.button("🗑️ Delete", key=f"x_{r['id']}"):
                    db.delete_review(r["id"])
                    st.rerun()

# ===================================================== VERIFY & DISCOVER
elif PAGE == "✅ Verify & Discover":
    st.header("✅ Verify & Discover")
    t1, t2, t3 = st.tabs(["Verify a reference / DOI", "Check a claim against my papers", "Discover related papers"])
    with t1:
        ref = st.text_input("DOI, DOI URL, or paper title / reference string")
        if st.button("Verify", disabled=not ref):
            with st.spinner("Querying OpenAlex, Crossref, Semantic Scholar…"):
                res = verify_reference(ref)
            icon = {"verified": "✅", "partial": "🟡", "retracted": "⛔", "not_found": "❌"}[res["verdict"]]
            st.subheader(f"{icon} {res['verdict'].replace('_', ' ').title()}")
            for r in res["results"]:
                with st.container(border=True):
                    st.markdown(f"**{r['source']}** — {r.get('title')}")
                    st.json({k: v for k, v in r.items() if v not in (None, "") and k != "source"})
        st.divider()
        st.caption("Batch-check the DOIs of papers in your library")
        if st.button("Check all library DOIs"):
            for p in db.list_papers():
                if p["doi"]:
                    res = verify_reference(p["doi"])
                    st.write(f"{p['title'][:80]} → **{res['verdict']}**")
    with t2:
        claim = st.text_area("Claim to verify")
        pids = paper_picker("verify")
        if st.button("Check claim", disabled=not claim):
            with st.spinner("Retrieving evidence and assessing…"):
                res = verify_claim(claim, get_gem(), S, pids)
            st.subheader(f"Verdict: {res.get('verdict')}  ·  confidence {res.get('confidence', '–')}")
            st.write(res.get("explanation", ""))
            hits = res.get("hits", [])
            for ev in res.get("evidence", []):
                try:
                    h = hits[int(ev.get("source")) - 1]
                    src = f"{h['title']}, p.{h.get('page')}"
                except Exception:
                    src = str(ev.get("source"))
                st.markdown(f"- *{ev.get('stance')}* — “{ev.get('quote')}” — **{src}**")
    with t3:
        dq = st.text_input("Topic to discover")
        if dq:
            with st.spinner("Searching OpenAlex…"):
                results = search_literature(dq, 15)
            for r in results:
                with st.container(border=True):
                    st.markdown(f"**{r['title']}** ({r['year']}) — {r['authors']}")
                    st.caption(f"{r.get('venue') or ''} · {r.get('citations')} citations"
                               + (f" · DOI {r['doi']}" if r["doi"] else ""))
                    if r.get("oa_url"):
                        st.markdown(f"[Open-access PDF]({r['oa_url']})")

# ================================================================= NOTES
elif PAGE == "🖍️ Notes & Highlights":
    st.header("🖍️ Notes & Highlights")
    papers = db.list_papers()
    if papers:
        with st.form("new_note", clear_on_submit=True):
            opts = {p["id"]: paper_label(p) for p in papers}
            pid = st.selectbox("Paper", list(opts), format_func=lambda x: opts[x])
            quote = st.text_area("Highlighted text / quote")
            note = st.text_area("Your note")
            a, b = st.columns(2)
            page = a.number_input("Page", 0, 10000, 0)
            color = b.selectbox("Color", ["yellow", "green", "blue", "pink"])
            if st.form_submit_button("Add note"):
                db.add_note(pid, quote, note, page or None, color)
                st.rerun()
    q = st.text_input("Search notes")
    dot = {"yellow": "🟨", "green": "🟩", "blue": "🟦", "pink": "🟪"}
    notes = db.list_notes(query=q)
    for n in notes:
        with st.container(border=True):
            st.markdown(f"{dot.get(n['color'], '🟨')} **{n['title']}**" + (f" — p.{n['page']}" if n["page"] else ""))
            if n["quote"]:
                st.markdown(f"> {n['quote']}")
            if n["note"]:
                st.write(n["note"])
            if st.button("Delete", key=f"dn_{n['id']}"):
                db.delete_note(n["id"])
                st.rerun()
    if notes:
        md = "\n\n".join(f"### {n['title']} (p.{n['page'] or '?'})\n> {n['quote']}\n\n{n['note']}" for n in notes)
        st.download_button("⬇️ Export notes (.md)", md, file_name="notes.md")

# ============================================================== SETTINGS
elif PAGE == "⚙️ Settings":
    st.header("⚙️ Settings")
    key = st.text_input("Google API key (Gemini)", S.google_api_key, type="password",
                        help="Get one at https://aistudio.google.com/apikey")
    if "model_lists" not in st.session_state:
        st.session_state.model_lists = (DEFAULT_CHAT_MODELS, DEFAULT_EMBED_MODELS)
    if st.button("🔄 Load available models from my key", disabled=not key):
        S.google_api_key = key
        with st.spinner("Listing models…"):
            st.session_state.model_lists = Gemini(S).list_models()
    chat_models, emb_models = st.session_state.model_lists
    chat_models = list(dict.fromkeys([S.chat_model, S.fallback_model] + list(chat_models)))
    emb_models = list(dict.fromkeys([S.embed_model] + list(emb_models)))
    c1, c2 = st.columns(2)
    chat_model = c1.selectbox("Chat model", chat_models, index=chat_models.index(S.chat_model))
    fallback = c2.selectbox("Fallback model (used if the main one fails)", chat_models,
                            index=chat_models.index(S.fallback_model))
    c1, c2 = st.columns(2)
    embed_model = c1.selectbox("Embedding model", emb_models, index=emb_models.index(S.embed_model))
    embed_dim = c2.selectbox("Embedding dimensions", [768, 1536, 3072], index=[768, 1536, 3072].index(S.embed_dim)
                             if S.embed_dim in (768, 1536, 3072) else 0)
    c1, c2, c3 = st.columns(3)
    temperature = c1.slider("Temperature", 0.0, 1.0, float(S.temperature), 0.05)
    max_tokens = c2.number_input("Max output tokens", 256, 65536, int(S.max_output_tokens), 256)
    top_k = c3.slider("Retrieved excerpts per question", 3, 30, int(S.top_k))
    c1, c2 = st.columns(2)
    lang = c1.selectbox("Answer language", ["English", "Hindi", "Marathi", "Vietnamese", "Japanese", "Spanish",
                                            "French", "German", "Chinese"],
                        index=["English", "Hindi", "Marathi", "Vietnamese", "Japanese", "Spanish", "French",
                               "German", "Chinese"].index(S.output_language)
                        if S.output_language in ["English", "Hindi", "Marathi", "Vietnamese", "Japanese", "Spanish",
                                                 "French", "German", "Chinese"] else 0)
    chunk_size = c2.slider("Chunk size (tokens, applies to new imports)", 200, 1500, int(S.chunk_size), 50)

    if st.button("💾 Save settings", type="primary"):
        changed_embed = embed_model != S.embed_model or embed_dim != S.embed_dim
        S.google_api_key, S.chat_model, S.fallback_model = key.strip(), chat_model, fallback
        S.embed_model, S.embed_dim, S.temperature = embed_model, int(embed_dim), float(temperature)
        S.max_output_tokens, S.top_k, S.output_language, S.chunk_size = int(max_tokens), int(top_k), lang, int(chunk_size)
        S.save()
        st.success("Saved.")
        if changed_embed and db.stats()["chunks"]:
            st.warning("Embedding model/size changed — use 'Re-index library' below so search keeps working.")
    if st.button("🧪 Test connection"):
        try:
            st.success(f"Gemini replied: {Gemini(S).test_connection()}")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    st.divider()
    if st.button("♻️ Re-index whole library (re-embed all papers)"):
        gem = get_gem()
        for p in db.list_papers():
            with st.spinner(f"Re-indexing {p['title'][:60]}…"):
                ingest(Path(p["file_path"]), S, gem, pid=p["id"], summarize=False, log=lambda *_: None)
        st.success("Re-index complete.")
