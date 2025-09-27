# scraper/ollama_rerank.py
import math
import httpx


def _cosine(a, b):
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    return 0.0 if (da == 0 or db == 0) else num / (da * db)


def embed(text: str, model: str = "nomic-embed-text", host: str = "http://127.0.0.1:11434"):
    # Ollama embeddings API
    url = f"{host}/api/embeddings"
    with httpx.Client(timeout=30) as c:
        r = c.post(url, json={"model": model, "prompt": text})
        r.raise_for_status()
        return r.json()["embedding"]


def rerank(query: str, docs, model: str = "nomic-embed-text", host: str = "http://127.0.0.1:11434"):
    """
    docs: list[dict] with 'text' (and 'title','url' optional)
    returns: list[(score, doc_dict)]
    """
    q_emb = embed(query, model=model, host=host)
    out = []
    for d in docs:
        # truncate to keep it snappy; adjust as needed
        txt = (d.get("title", "") + "\n" + d.get("text", ""))[:8000]
        d_emb = embed(txt, model=model, host=host)
        out.append((_cosine(q_emb, d_emb), d))
    out.sort(key=lambda x: x[0], reverse=True)
    return out
