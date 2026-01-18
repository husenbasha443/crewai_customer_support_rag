SIMILARITY_THRESHOLD = 0.75

def retrieve_or_fallback(query, vector_db):
    results = vector_db.similarity_search_with_score(query, k=2)

    if results:
        doc, score = results[0]
        if score < SIMILARITY_THRESHOLD:
            return {"source": "vector_db", "context": doc.page_content}

    return {"source": "llm", "context": ""}
