"""
RAG chunking comparison on Tiny Shakespeare using LlamaIndex
Comments avoid the hyphen character by request
"""

import time
import numpy as np
import pandas as pd
import requests

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import (
    TokenTextSplitter,
    SemanticSplitterNodeParser,
    SentenceWindowNodeParser,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext


TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def fetch_corpus(url: str) -> str:
    """Download the Tiny Shakespeare text"""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.text


def build_embed_model():
    """Create a HuggingFace embedding model"""
    return HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)


def chunk_documents(docs, method: str, embed_model):
    """Return nodes for a given chunking method"""
    method = method.lower()
    if method == "token":
        splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=32)
        nodes = splitter.get_nodes_from_documents(docs)
        return nodes
    if method == "semantic":
        splitter = SemanticSplitterNodeParser(buffer_size=3, embed_model=embed_model)
        nodes = splitter.get_nodes_from_documents(docs)
        return nodes
    if method == "sentence_window":
        splitter = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window"
        )
        nodes = splitter.get_nodes_from_documents(docs)
        return nodes
    raise ValueError(f"Unknown method: {method}")


def build_index(nodes, embed_model):
    """Create an in memory vector store and build the index"""
    storage_ctx = StorageContext.from_defaults(vector_store=SimpleVectorStore())
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_ctx,
        embed_model=embed_model,
        show_progress=True
    )
    return index


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two one dimensional numpy arrays"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_only(index, nodes, query: str, k: int, embed_model):
    """Run retrieval for a single query and print the required information"""
    retriever = index.as_retriever(similarity_top_k=k)

    # query vector and first 8 values
    q_vec = embed_model.get_text_embedding(query)
    q_vec = np.array(q_vec, dtype=np.float32)
    print()
    print("Query embedding shape:", q_vec.shape)
    print("First 8 values:", list(map(float, q_vec[:8])))

    # time the retriever
    t0 = time.time()
    results = retriever.retrieve(query)
    latency_ms = (time.time() - t0) * 1000.0

    rows = []
    cosines = []
    doc_vecs = []

    for rank, res in enumerate(results, start=1):
        # result content
        text = str(res.get_content())
        # store score if present
        store_score = None
        if hasattr(res, "score") and res.score is not None:
            try:
                store_score = float(res.score)
            except Exception:
                store_score = None

        d_vec = embed_model.get_text_embedding(text)
        d_vec = np.array(d_vec, dtype=np.float32)
        doc_vecs.append(d_vec)
        c = cosine_similarity(q_vec, d_vec)
        cosines.append(c)

        rows.append({
            "rank": rank,
            "store_score": None if store_score is None else round(store_score, 4),
            "cosine_sim": round(c, 4),
            "chunk_len": len(text),
            "preview": text[:160].replace("\n", " ")
        })

    if len(doc_vecs) > 0:
        stacked = np.stack(doc_vecs, axis=0)
    else:
        stacked = np.zeros((0, q_vec.shape[0]), dtype=np.float32)

    print("Stacked doc vectors shape:", stacked.shape)

    # summary metrics
    top1 = max(cosines) if cosines else None
    mean_k = float(np.mean(cosines)) if cosines else None
    num_chunks = len(nodes)
    avg_len = float(np.mean([len(str(n.get_content())) for n in nodes])) if nodes else 0.0

    print(pd.DataFrame(rows))

    metrics = {
        "Top-1 Cosine": None if top1 is None else round(float(top1), 4),
        "Mean@k Cosine": None if mean_k is None else round(mean_k, 4),
        "#Chunks": int(num_chunks),
        "Avg Chunk Length": round(avg_len, 1),
        "Latency (ms)": round(float(latency_ms), 2)
    }
    return rows, metrics


def run_all(query: str, k: int = 3, save_prefix: str = "retrieval"):
    """Run the three pipelines and print a summary"""
    text = fetch_corpus(TINY_SHAKESPEARE_URL)
    docs = [Document(text=text)]
    embed_model = build_embed_model()

    all_metrics = {}
    all_rows = {}

    # token based
    print("\n=== Token based Chunking ===")
    token_nodes = chunk_documents(docs, "token", embed_model)
    token_index = build_index(token_nodes, embed_model)
    token_rows, token_metrics = retrieve_only(token_index, token_nodes, query, k, embed_model)
    all_metrics["Token"] = token_metrics
    all_rows["Token"] = token_rows

    # semantic
    print("\n=== Semantic Chunking ===")
    sem_nodes = chunk_documents(docs, "semantic", embed_model)
    sem_index = build_index(sem_nodes, embed_model)
    sem_rows, sem_metrics = retrieve_only(sem_index, sem_nodes, query, k, embed_model)
    all_metrics["Semantic"] = sem_metrics
    all_rows["Semantic"] = sem_rows

    # sentence window
    print("\n=== Sentence Window Chunking ===")
    sw_nodes = chunk_documents(docs, "sentence_window", embed_model)
    sw_index = build_index(sw_nodes, embed_model)
    sw_rows, sw_metrics = retrieve_only(sw_index, sw_nodes, query, k, embed_model)
    all_metrics["SentenceWindow"] = sw_metrics
    all_rows["SentenceWindow"] = sw_rows

    # comparison table
    df = pd.DataFrame(all_metrics).T
    print("\n=== Retrieval Quality Comparison ===")
    print(df)

    # optional writes to csv for your report
    df.to_csv(f"{save_prefix}_quality_summary.csv", index=True)
    pd.DataFrame(all_rows["Token"]).to_csv(f"{save_prefix}_token_rows.csv", index=False)
    pd.DataFrame(all_rows["Semantic"]).to_csv(f"{save_prefix}_semantic_rows.csv", index=False)
    pd.DataFrame(all_rows["SentenceWindow"]).to_csv(f"{save_prefix}_sentence_window_rows.csv", index=False)

    return df, all_rows


if __name__ == "__main__":
    # required query for this assignment
    print("=== RAG Chunking Comparison on Tiny Shakespeare ===")
    print("Testing query: 'Who are the two feuding houses?'")
    
    query = "Who are the two feuding houses?"
    df, all_rows = run_all(query=query, k=3)
    
    print("\n=== Analysis Summary ===")
    print("Results saved to CSV files for detailed analysis.")
    
    # Optional additional queries to strengthen comparison
    print("\n=== Additional Query Testing ===")
    print("Testing query: 'Who is Romeo in love with?'")
    df2, _ = run_all(query="Who is Romeo in love with?", k=3, save_prefix="romeo_query")
    
    print("\n=== Final Comparison ===")
    print("Primary Query Results:")
    print(df)
    print("\nSecondary Query Results:")
    print(df2)
