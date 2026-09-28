from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIR = "db/chroma"

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

PROMPT = """Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}"""


def to_text(content):
    if isinstance(content, str):
        return content
    return "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in content
    )


def ask(question: str):
    docs = retriever.invoke(question)
    print(f"\n[retrieved {len(docs)} chunks]")
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    answer = llm.invoke(PROMPT.format(context=context, question=question))

    print("\nAnswer:\n", to_text(answer.content))
    print("\nSources:")
    seen = set()
    for d in docs:
        page = d.metadata.get("page")
        label = f"{d.metadata.get('source')}" + (f" (page {page + 1})" if page is not None else "")
        if label not in seen:
            seen.add(label)
            print(" -", label)


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip().strip('"')
        if q.lower() in {"exit", "quit", ""}:
            break
        ask(q)