"""Extract plain text from uploaded files (PDF / TXT / Markdown)."""
import io

import pdfplumber


def extract_text(filename: str, raw: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _extract_pdf(raw)
    if name.endswith((".txt", ".md", ".markdown")):
        return raw.decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _extract_pdf(raw: bytes) -> str:
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages)
