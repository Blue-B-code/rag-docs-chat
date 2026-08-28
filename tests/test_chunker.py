from app.rag.chunker import chunk_text


def test_short_text_is_single_chunk():
    assert chunk_text("Hello world") == ["Hello world"]


def test_empty_and_whitespace_text():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_long_text_splits_into_multiple_chunks_within_limit():
    text = "word " * 500  # ~2500 chars, no natural boundaries
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_all_words_preserved_across_chunks():
    words = [f"sentence{i}" for i in range(50)]
    text = ". ".join(words) + "."
    chunks = chunk_text(text, chunk_size=120, overlap=30)
    joined = " ".join(chunks)
    for word in words:
        assert word in joined


def test_respects_chunk_size():
    text = "Lorem ipsum dolor sit amet. " * 60
    chunks = chunk_text(text, chunk_size=300, overlap=60)
    assert all(len(c) <= 300 for c in chunks)
