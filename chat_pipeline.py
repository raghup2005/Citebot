from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIR = "db/chroma"

# --- Same setup as retrieval_pipeline.py ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

# --- Memory: a simple list of (question, answer) pairs ---python chat_pipeline.py w
history = []

REWRITE_PROMPT = """Below is a chat history and a follow-up question.
Rewrite the follow-up so it makes sense on its own, without the history.
If it already makes sense on its own, return it unchanged.
Return ONLY the rewritten question.

Chat history:
{history}

Follow-up question: {question}"""
ANSWER_PROMPT = """Answer the question using only the context below.
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


def format_history():
    # Only keep the last 5 exchanges so the prompt stays small
    recent = history[-5:]
    return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in recent)


def rewrite_question(question):
    if not history:
        return question  # first question: nothing to rewrite
    prompt = REWRITE_PROMPT.format(history=format_history(), question=question)
    return to_text(llm.invoke(prompt).content).strip()


def ask(question):
    standalone = rewrite_question(question)
    if standalone != question:
        print(f"[searching for: {standalone}]")

    docs = retriever.invoke(standalone)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    answer = to_text(
        llm.invoke(ANSWER_PROMPT.format(context=context, question=standalone)).content
    )

    print("\nAnswer:\n", answer)
    print("\nSources:")
    for label in sorted({d.metadata.get("source") for d in docs}):
        print(" -", label)

    history.append((question, answer))


if __name__ == "__main__":
    print("Chat with your documents. Type 'reset' to clear memory, 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip().strip('"')
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "reset":
            history.clear()
            print("Memory cleared.")
            continue
        if q:
            ask(q)