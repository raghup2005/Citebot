# Chat with Your Documents (RAG)

A retrieval-augmented generation app that answers questions about a set of
company documents, with sources and conversation memory.

## How it works
1. **Ingest:** load PDFs/text files, split into chunks, embed with Gemini, store in Chroma
2. **Retrieve:** find the 4 most relevant chunks for a question
3. **Answer:** Gemini answers using only those chunks and lists its sources
4. **Memory:** follow-up questions ("Who founded it?") are rewritten into
   standalone questions before searching

## Tech stack
Python, LangChain, ChromaDB, Google Gemini (embeddings + chat), Gradio

## Setup
```
pip install -r requirements.txt
```
Create a `.env` file:
```
GOOGLE_API_KEY=your_key_here
```
Put `.pdf` or `.txt` files in `docs/`, then:
```
python ingestion_pipeline.py
python app_gradio.py
```

## Reliability
Free-tier APIs can return 503/429 errors, so the app retries and falls
back to a second model automatically.

## Example questions
- Where is Apple headquartered?
- Who founded it?
- Compare Flipkart and Amazon India.
