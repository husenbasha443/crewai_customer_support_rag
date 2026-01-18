from pathlib import Path

from customer_support_rag.tools.pdf_loader import load_pdfs
from customer_support_rag.tools.vector_store import build_or_load_index
from customer_support_rag.tools.rag_router import retrieve_or_fallback
from customer_support_rag.crew import CustomerSupportCrew

from langchain_community.chat_models import ChatOllama


def run():
    pdf_folder = Path("customer_support_rag/data/uploads")

    texts = load_pdfs(pdf_folder)
    vector_db = build_or_load_index(texts)

    user_query = input("Ask a question: ")

    route = retrieve_or_fallback(user_query, vector_db)

    # LLM fallback
    if route["source"] == "llm":
        llm = ChatOllama(model="llama3.1:8b")
        route["context"] = llm.invoke(user_query).content

    inputs = {
        "user_query": user_query,
        "context": route["context"],
        "source": route["source"],
    }

    CustomerSupportCrew().crew().kickoff(inputs=inputs)


if __name__ == "__main__":
    run()
