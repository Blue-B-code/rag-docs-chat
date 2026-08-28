"""Recursive text chunking with overlap (no external dependencies)."""


def _best_boundary(text: str, lo: int, hi: int) -> int | None:
    """Find the last paragraph/sentence boundary in [lo, hi), coarser first."""
    for sep in ("\n\n", "\n", ". ", "! ", "? "):
        idx = text.rfind(sep, lo, hi)
        if idx != -1:
            return idx + len(sep)
    return None


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into overlapping chunks, preferring sentence/paragraph breaks."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    n = len(text)
    chunks: list[str] = []
    start = 0
    min_boundary = chunk_size // 2
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            boundary = _best_boundary(text, start + min_boundary, end)
            if boundary is not None:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks
