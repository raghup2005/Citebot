import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = "docs"
PERSIST_DIR = "db/chroma"


def load_documents(docs_dir: str):
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Create a '{docs_dir}' folder and put your files in it.")

    documents = []
    for name in os.listdir(docs_dir):
        path = os.path.join(docs_dir, name)
        lower = name.lower()
        if lower.endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())
        elif lower.endswith(".txt"):
            documents.extend(TextLoader(path, encoding="utf-8").load())
    return documents


def main():
    documents = load_documents(DOCS_DIR)
    if not documents:
        raise ValueError("No .pdf or .txt files found in docs/")
    print(f"Loaded {len(documents)} pages/documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"Saved vector store to {PERSIST_DIR}")


if __name__ == "__main__":
    main()