from document_engine import chunk_pages, retrieve_chunks


def test_chunk_pages_preserves_metadata():
    pages = [{"source": "sample.pdf", "page": 2, "text": "alpha beta gamma delta epsilon"}]
    chunks = chunk_pages(pages, chunk_size=3, overlap=1)
    assert len(chunks) == 2
    assert chunks[0]["source"] == "sample.pdf"
    assert chunks[0]["page"] == 2
    assert chunks[1]["text"] == "gamma delta epsilon"


def test_retrieve_chunks_returns_relevant_passage():
    chunks = [
        {"source": "a.pdf", "page": 1, "text": "The project budget is ten thousand dollars."},
        {"source": "b.pdf", "page": 3, "text": "The office cafeteria serves lunch."},
    ]
    results = retrieve_chunks("What is the project budget?", chunks, top_k=1)
    assert results
    assert results[0]["source"] == "a.pdf"
