"""PDF extraction, chunking, retrieval, and optional LLM answer generation."""
from io import BytesIO
from typing import Dict, List
import os
import re

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_pdf_pages(pdf_bytes: bytes, filename: str) -> List[Dict]:
    """Extract selectable text from each PDF page."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"source": filename, "page": page_number, "text": text})
    return pages


def chunk_pages(pages: List[Dict], chunk_size: int = 900, overlap: int = 150) -> List[Dict]:
    """Split page text into overlapping word chunks while retaining page metadata."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Use chunk_size > overlap >= 0.")
    chunks = []
    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            if chunk_text.strip():
                chunks.append({"source": page["source"], "page": page["page"], "text": chunk_text})
            if end == len(words):
                break
            start = end - overlap
    return chunks


def retrieve_chunks(question: str, chunks: List[Dict], top_k: int = 4) -> List[Dict]:
    """Rank document chunks using TF-IDF cosine similarity."""
    if not question.strip() or not chunks:
        return []
    corpus = [item["text"] for item in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(corpus + [question])
    except ValueError:
        return []
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = scores.argsort()[::-1][:top_k]
    results = []
    for index in ranked:
        if scores[index] <= 0:
            continue
        item = dict(chunks[index])
        item["score"] = float(scores[index])
        results.append(item)
    return results


def answer_with_openai(question: str, matches: List[Dict]) -> str:
    """Generate a grounded answer using OpenAI, when an API key is configured."""
    from openai import OpenAI

    context = "\n\n".join(
        f"[Source: {m['source']}, page {m['page']}]\n{m['text']}" for m in matches
    )
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an enterprise document assistant. Answer only from the supplied "
                    "document excerpts. If the excerpts do not contain the answer, say so. "
                    "Cite claims using [filename, page N]. Do not invent facts."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nDocument excerpts:\n{context}"},
        ],
    )
    return response.choices[0].message.content or "No answer was generated."
