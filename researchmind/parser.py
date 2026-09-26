"""Document parsing: PDF, DOCX, EPUB, TXT, MD, HTML -> {page_number: text}.
Scanned PDF pages (no text layer) can be OCR'd with Gemini vision."""

from __future__ import annotations

import re
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".epub", ".txt", ".md", ".markdown", ".html", ".htm"}
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


def parse_file(path: Path, ocr=None, ocr_progress=None) -> tuple[dict[int, str], dict]:
    """Returns (text_by_page, metadata). `ocr` is an optional callable(bytes)->str."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(path, ocr, ocr_progress)
    if ext == ".docx":
        return _parse_docx(path)
    if ext == ".epub":
        return _parse_epub(path)
    if ext in (".html", ".htm"):
        return _parse_html(path.read_text(encoding="utf-8", errors="ignore"))
    if ext in (".txt", ".md", ".markdown"):
        return {1: path.read_text(encoding="utf-8", errors="ignore")}, {}
    raise ValueError(f"Unsupported file type: {ext}")


def _parse_pdf(path: Path, ocr, ocr_progress):
    import pymupdf

    doc = pymupdf.open(path)
    pages: dict[int, str] = {}
    meta = {k: v for k, v in (doc.metadata or {}).items() if v}
    scanned = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        if len(text.strip()) < 30:
            scanned.append(i)
        pages[i] = text
    # OCR pages without a text layer
    if ocr and scanned:
        for n, i in enumerate(scanned, 1):
            try:
                pix = doc[i - 1].get_pixmap(dpi=150)
                pages[i] = ocr(pix.tobytes("png"))
            except Exception as e:  # noqa: BLE001
                pages[i] = pages[i] or f"[OCR failed on page {i}: {e}]"
            if ocr_progress:
                ocr_progress(n, len(scanned))
    out_meta = {
        "title": meta.get("title", ""),
        "authors": meta.get("author", ""),
        "num_pages": len(doc),
        "scanned_pages": len(scanned),
    }
    doc.close()
    return pages, out_meta


def _parse_docx(path: Path):
    import docx

    d = docx.Document(str(path))
    paras = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            paras.append(" | ".join(c.text.strip() for c in row.cells))
    cp = d.core_properties
    return {1: "\n\n".join(paras)}, {"title": cp.title or "", "authors": cp.author or ""}


def _parse_epub(path: Path):
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(path))
    pages = {}
    n = 0
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        text, _ = _parse_html(item.get_content().decode("utf-8", errors="ignore"))
        if text[1].strip():
            n += 1
            pages[n] = text[1]
    title = (book.get_metadata("DC", "title") or [[""]])[0][0]
    creator = (book.get_metadata("DC", "creator") or [[""]])[0][0]
    return pages, {"title": title, "authors": creator}


def _parse_html(html: str):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer"]):
        t.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return {1: text}, {"title": title}


def guess_metadata(pages: dict[int, str], meta: dict, file_name: str) -> dict:
    """Heuristic title / DOI / year from the first page when PDF metadata is empty."""
    first = "\n".join(pages.get(p, "") for p in sorted(pages)[:2])
    title = (meta.get("title") or "").strip()
    if not title or len(title) < 5 or title.lower().startswith(("microsoft", "untitled")):
        lines = [ln.strip() for ln in first.splitlines() if 15 < len(ln.strip()) < 200]
        title = lines[0] if lines else Path(file_name).stem
    doi = ""
    m = DOI_RE.search(first)
    if m:
        doi = m.group(1).rstrip(".,;)")
    year = None
    ym = YEAR_RE.findall(first[:3000])
    if ym:
        year = int(max(ym, key=ym.count))
    return {"title": title[:300], "authors": (meta.get("authors") or "")[:300], "doi": doi, "year": year,
            "num_pages": meta.get("num_pages", len(pages))}
