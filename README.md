# Enterprise GenAI Document Assistant

A beginner-friendly AI portfolio application for PDF question answering. It extracts PDF text, creates overlapping chunks, retrieves relevant passages with TF-IDF/cosine similarity, and optionally generates answers using an LLM grounded in the retrieved excerpts.

## Features

- Upload multiple text-based PDFs (20 MB combined limit).
- Extract text page-by-page and retain source metadata.
- Split text into overlapping chunks.
- Retrieve relevant passages for a natural-language question.
- Optional OpenAI answer generation with source/page citations.
- Evidence-only mode works without an API key.
- Unit tests for chunking and retrieval.

## Tech stack

Python 3.11, Streamlit, pypdf, scikit-learn, OpenAI Python SDK (optional), python-dotenv, pytest.

## Run in GitHub Codespaces

1. Open this repository on GitHub.
2. Select **Code → Codespaces → Create codespace on main**.
3. In the terminal run:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py --server.address 0.0.0.0
   ```

4. Open the forwarded port **8501**.

The included `.devcontainer/devcontainer.json` installs dependencies and forwards port 8501 when the Codespace is created.

## Optional OpenAI setup

1. Create an API key in your OpenAI developer account.
2. In GitHub, go to **Settings → Secrets and variables → Codespaces**.
3. Add a secret named `OPENAI_API_KEY`.
4. Stop and restart the Codespace.
5. Turn on **Use OpenAI to generate answers** in the sidebar.

Never commit an API key. API usage may incur charges. The app works in evidence-only mode without a key. The model defaults to `gpt-4o-mini`; set `OPENAI_MODEL` to change it.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest -q
```

## Architecture

```text
PDF upload
  -> pypdf page text extraction
  -> overlapping chunks with filename/page metadata
  -> TF-IDF vectorization + cosine similarity
  -> top relevant passages
  -> evidence-only display OR optional grounded LLM response
```

## Demo suggestions

Use a public annual report, public product manual, or your own non-confidential project report. Ask:
- What are the main strategic priorities?
- Which risks are mentioned?
- Which page discusses the implementation timeline?

Show the answer and the retrieved source passages.

## Limitations and responsible use

- Scanned/image-only PDFs need OCR; OCR is not included.
- Retrieval is a lightweight TF-IDF baseline, not a production vector database.
- No authentication, persistent document storage, or multi-user access controls.
- Do not upload confidential, sensitive, or personal documents.
- Verify all outputs before using them for business decisions.

## Future improvements

Add OCR, embedding-based retrieval with FAISS/Chroma, evaluation metrics for retrieval and faithfulness, authentication, document deletion, logging, rate limits, and privacy controls.

## Interview explanation

"I built a PDF question-answering assistant. It extracts text while retaining page metadata, chunks it with overlap, retrieves relevant passages using TF-IDF and cosine similarity, and can optionally pass only those passages to an LLM. It cites source pages and provides an evidence-only mode when no API key is configured. I also added upload limits and unit tests."

Be prepared to explain chunking, overlap, TF-IDF, cosine similarity, grounding, evaluation, and production limitations.
