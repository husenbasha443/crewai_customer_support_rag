import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from customer_support_rag.tools.embedder import get_embeddings

VECTOR_PATH = "customer_support_rag/vector_db/faiss_index"


def build_or_load_index(texts):
    embeddings = get_embeddings()

    if os.path.exists(VECTOR_PATH):
        return FAISS.load_local(
            VECTOR_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    docs = [Document(page_content=t) for t in texts if t]
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_PATH)
    return db
