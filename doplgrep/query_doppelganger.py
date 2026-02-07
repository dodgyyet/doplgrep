"""Database query functions for face similarity search."""
import sqlite3
import numpy as np
from pathlib import Path
from .utils import cosine_similarity


def query_database_hnsw(db_path: str, query_embedding: np.ndarray, top_n: int = 1) -> list:
    """
    Query using HNSW index for fast search.
    
    Args:
        db_path: Path to SQLite database
        query_embedding: Query embedding vector
        top_n: Number of top results to return
        
    Returns:
        List of (similarity_score, image_path) tuples
    """
    index_path = db_path.replace('.sqlite', '.hnsw')
    
    if Path(index_path).exists():
        print("⚡ Using HNSW index for fast search...")
        from .vector_index import HNSWIndex
        
        index = HNSWIndex()
        index.load(index_path, db_path)
        return index.query(query_embedding.reshape(1, -1), top_k=top_n)
    else:
        print("No HNSW index found, using brute-force search...")
        print(f"For faster search on large datasets, run:")
        print(f"doplgrep --mkidx {db_path}\n")
        return query_database_bruteforce(db_path, query_embedding, top_n)


def query_database_bruteforce(db_path: str, query_embedding: np.ndarray, top_n: int = 1) -> list:
    """
    Fallback brute-force query for small datasets or when HNSW unavailable.
    
    Args:
        db_path: Path to SQLite database
        query_embedding: Query embedding vector
        top_n: Number of top results to return
        
    Returns:
        List of (similarity_score, image_path) tuples
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT image_path, embedding FROM face_embeddings")
    
    results = []
    for row in cursor.fetchall():
        img_path, blob = row
        db_embedding = np.frombuffer(blob, dtype=np.float32)
        
        similarity = cosine_similarity(query_embedding, db_embedding)
        results.append((similarity, img_path))
    
    conn.close()
    
    # Sort by similarity (descending)
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top_n]