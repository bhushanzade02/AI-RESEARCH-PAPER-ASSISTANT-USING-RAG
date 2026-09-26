"""Hybrid retrieval: SQLite FTS5 (BM25) + Gemini-embedding cosine similarity,
fused with Reciprocal Rank Fusion, then diversified per paper."""

from __future__ import annotations

import numpy as np

from . import db
from .gemini import Gemini


def hybrid_search(query: str, gem: Gemini | None, paper_ids: list[str] | None = None,
                  top_k: int = 8, max_per_paper: int | None = None) -> list[dict]:
    rows = db.get_chunks(paper_ids, with_embeddings=True)
    if not rows:
        return []
    by_id = {r["id"]: r for r in rows}

    # --- lexical (BM25)
    bm25 = db.fts_search(query, paper_ids, limit=60)
    bm25_rank = {cid: i for i, (cid, _) in enumerate(bm25)}

    # --- semantic
    vec_rank: dict[int, int] = {}
    vec_score: dict[int, float] = {}
    if gem is not None:
        with_vec = [r for r in rows if r["vec"] is not None]
        if with_vec:
            try:
                q = gem.embed_query(query)
                mat = np.vstack([r["vec"] for r in with_vec])
                if mat.shape[1] == q.shape[0]:
                    sims = mat @ q
                    order = np.argsort(-sims)[:60]
                    for rank, j in enumerate(order):
                        cid = with_vec[int(j)]["id"]
                        vec_rank[cid] = rank
                        vec_score[cid] = float(sims[int(j)])
            except Exception:
                pass  # fall back to lexical only

    # --- RRF fusion
    k = 60
    fused: dict[int, float] = {}
    for cid, r in bm25_rank.items():
        fused[cid] = fused.get(cid, 0) + 1 / (k + r)
    for cid, r in vec_rank.items():
        fused[cid] = fused.get(cid, 0) + 1 / (k + r)
    if not fused:  # nothing matched: return first chunks of each paper
        seen, out = set(), []
        for r in rows:
            if r["paper_id"] not in seen:
                seen.add(r["paper_id"])
                out.append({**r, "score": 0.0})
        return [_strip(o) for o in out[:top_k]]

    ranked = sorted(fused.items(), key=lambda x: -x[1])
    per_paper: dict[str, int] = {}
    out = []
    if max_per_paper is None:
        n_papers = len({r["paper_id"] for r in rows})
        max_per_paper = max(2, top_k // max(1, min(n_papers, top_k)) + 1) if n_papers > 1 else top_k
    for cid, score in ranked:
        r = by_id.get(cid)
        if not r or r.get("section") == "References":
            continue
        if per_paper.get(r["paper_id"], 0) >= max_per_paper:
            continue
        per_paper[r["paper_id"]] = per_paper.get(r["paper_id"], 0) + 1
        out.append({**r, "score": score, "similarity": vec_score.get(cid)})
        if len(out) >= top_k:
            break
    return [_strip(o) for o in out]


def _strip(r: dict) -> dict:
    r = dict(r)
    r.pop("vec", None)
    return r


def format_context(hits: list[dict], max_chars: int = 24000) -> str:
    parts, total = [], 0
    for i, h in enumerate(hits, 1):
        head = f"[{i}] Document: {h['title']}" + (f" ({h['year']})" if h.get("year") else "")
        head += f" | page {h['page']}" if h.get("page") else ""
        if h.get("section"):
            head += f" | section: {h['section']}"
        block = f"{head}\n{h['content']}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts)
