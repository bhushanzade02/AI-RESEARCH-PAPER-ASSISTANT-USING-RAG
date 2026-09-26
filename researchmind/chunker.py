"""Paragraph-aware chunker with overlap, keeping page numbers and detected
section headers so answers can cite [Title, page X]."""

from __future__ import annotations

import re

HEADER_RE = re.compile(
    r"^\s*(\d+(\.\d+)*\.?\s+)?(abstract|introduction|background|related work|method(s|ology)?|approach|"
    r"experiments?|results?|evaluation|discussion|limitations?|conclusions?|future work|references)\b",
    re.IGNORECASE,
)


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _clean(text: str) -> str:
    text = text.replace("­", "")
    text = re.sub(r"-\n(\w)", r"\1", text)          # de-hyphenate line breaks
    text = re.sub(r"(?<![.\n:])\n(?!\n)", " ", text)  # join wrapped lines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def chunk_pages(text_by_page: dict[int, str], chunk_size: int = 500, overlap: int = 80) -> list[dict]:
    chunks: list[dict] = []
    section = ""
    buf, buf_page = "", None
    overlap_chars = overlap * 4

    def flush():
        nonlocal buf, buf_page
        if buf.strip() and count_tokens(buf) > 20:
            chunks.append({"content": buf.strip(), "page": buf_page, "section": section})
        tail = buf[-overlap_chars:] if overlap else ""
        buf = tail[tail.find(" ") + 1:] if " " in tail else tail
        buf_page = None

    for page in sorted(text_by_page):
        # keep citations page-accurate: start a new chunk at page breaks once the buffer is substantial
        if buf_page is not None and count_tokens(buf) >= chunk_size // 3:
            flush()
        for para in _split_paragraphs(_clean(text_by_page[page])):
            if len(para) < 120 and HEADER_RE.match(para):
                section = para[:80]
            if re.match(r"^\s*references\s*$", para, re.I):
                section = "References"
            pieces = [para]
            if count_tokens(para) > chunk_size:
                pieces, cur = [], ""
                for s in _split_sentences(para):
                    if count_tokens(cur + s) > chunk_size and cur:
                        pieces.append(cur)
                        cur = ""
                    cur += s + " "
                if cur:
                    pieces.append(cur)
            for piece in pieces:
                if buf_page is None:
                    buf_page = page
                if count_tokens(buf + piece) > chunk_size and count_tokens(buf) > chunk_size // 2:
                    flush()
                    buf_page = page
                buf += ("\n\n" if buf else "") + piece
    flush()
    return chunks
