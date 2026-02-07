"""HNSW vector index for fast similarity search."""
import hnswlib
import numpy as np
import sqlite3
from pathlib import Path


class HNSWIndex:
    """Fast vector similarity search using HNSW index."""
    
    def __init__(self, dim: int = 1024, max_elements: int = 200000):
        """
        Initialize HNSW index.
        
        Args:
            dim: Embedding dimension (1024 for DINOv3 ViT-L)
            max_elements: Maximum number of vectors
        """
        self.dim = dim
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=200,  # Higher = better quality, slower build
            M=16  # Higher = better recall, more memory
        )
        self.index.set_ef(50)  # Higher = better recall, slower search
        self.id_to_path = {}
    
    def build_from_database(self, db_path: str):
        """Build HNSW index from SQLite database."""
        print("Loading embeddings from database...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, image_path, embedding FROM face_embeddings")
        
        embeddings = []
        ids = []
        
        for row in cursor.fetchall():
            db_id, img_path, blob = row
            embedding = np.frombuffer(blob, dtype=np.float32)
            
            embeddings.append(embedding)
            ids.append(db_id)
            self.id_to_path[db_id] = img_path
        
        conn.close()
        
        print(f"Building HNSW index for {len(embeddings)} vectors...")
        embeddings_np = np.array(embeddings)
        self.index.add_items(embeddings_np, ids)
        print("✓ HNSW index built!")
    
    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> list:
        """
        Query index for nearest neighbors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            
        Returns:
            List of (similarity, image_path) tuples
        """
        # HNSW returns distances, convert to similarity
        labels, distances = self.index.knn_query(query_embedding, k=top_k)
        
        results = []
        for label, dist in zip(labels[0], distances[0]):
            similarity = 1 - dist  # Convert cosine distance to similarity
            img_path = self.id_to_path[label]
            results.append((similarity, img_path))
        
        return results
    
    def save(self, path: str):
        """Save index to disk."""
        self.index.save_index(path)
        print(f"✓ Index saved to {path}")
    
    def load(self, path: str, db_path: str):
        """Load index from disk."""
        self.index.load_index(path)
        
        # Reload id_to_path mapping
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, image_path FROM face_embeddings")
        self.id_to_path = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        print(f"✓ Index loaded from {path}")