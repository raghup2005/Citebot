import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIR = "db/chroma"

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
primary = ChatGoogleGenerativeAI(model="gemini-3.6-flash", timeout=60, max_retries=3)
backup = ChatGoogleGenerativeAI(model="gemini-3.7-flash", timeout=60, max_retries=2)
llm = primary.with_fallbacks([backup])

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
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def history_to_text(history):
    """Gradio versions send history in different formats; handle both."""
    lines = []
    for item in history[-10:]:
        if isinstance(item, dict):  # newer format: {"role": ..., "content": ...}
            who = "User" if item["role"] == "user" else "Assistant"
            lines.append(f"{who}: {to_text(item['content'])}")
        else:  # older format: [user_message, assistant_message]
            user_msg, bot_msg = item
            lines.append(f"User: {user_msg}")
            if bot_msg:
                lines.append(f"Assistant: {bot_msg}")
    return "\n".join(lines)


def chat(message, history):
    try:
        standalone = message
        if history:
            prompt = REWRITE_PROMPT.format(history=history_to_text(history), question=message)
            standalone = to_text(llm.invoke(prompt).content).strip()

        docs = retriever.invoke(standalone)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        answer = to_text(
            llm.invoke(ANSWER_PROMPT.format(context=context, question=standalone)).content
        )
        sources = sorted({d.metadata.get("source", "unknown") for d in docs})
        return f"{answer}\n\n**Sources:** " + ", ".join(sources)
    except Exception as e:
        return f"Something went wrong: {e}"


demo = gr.ChatInterface(
    fn=chat,
    title="Chat with your documents",
    description="Ask about Apple, Flipkart, Netflix, Nvidia, Microsoft, Amazon, Tesla or TCS.",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)