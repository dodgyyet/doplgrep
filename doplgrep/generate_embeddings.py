"""Generate embeddings for a dataset and store in SQLite."""
import sqlite3
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .image_embedder import ImageEmbedder
from .utils import preprocess_image


def create_vector_table(db_path: str) -> sqlite3.Connection:
    """Create SQLite table for storing embeddings."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create main table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL UNIQUE,
            embedding BLOB NOT NULL
        )
    """)
    
    conn.commit()
    return conn


def generate_embeddings(image_dir: str, db_path: str) -> None:
    """
    Generate embeddings for all images in directory.
    
    Args:
        image_dir: Directory containing images
        db_path: Output SQLite database path
    """
    print("Initializing DINOv3 embedder...")
    embedder = ImageEmbedder()
    
    print(f"Creating database at {db_path}...")
    conn = create_vector_table(db_path)
    cursor = conn.cursor()
    
    # Find all images
    image_paths = list(Path(image_dir).glob("**/*.png")) + \
                  list(Path(image_dir).glob("**/*.jpg")) + \
                  list(Path(image_dir).glob("**/*.jpeg"))
    
    print(f"Processing {len(image_paths)} images...")
    
    successful = 0
    for img_path in tqdm(image_paths):
        try:
            # Load and embed image
            pil_image = preprocess_image(str(img_path))
            embedding = embedder.embed(pil_image)
            
            # Convert to numpy and normalize
            embedding_np = embedding.cpu().numpy().flatten()
            embedding_np = embedding_np / np.linalg.norm(embedding_np)
            
            # Store in database
            cursor.execute(
                "INSERT OR REPLACE INTO face_embeddings (image_path, embedding) VALUES (?, ?)",
                (str(img_path), embedding_np.tobytes())
            )
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue
    
    conn.commit()
    conn.close()
    print(f"\nSuccessfully embedded {successful}/{len(image_paths)} images!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python generate_embeddings.py <image_dir> <output_db>")
        print("Example: python generate_embeddings.py ../data/ffhq_demo ../embeddings/ffhq_demo_embeds.sqlite")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    db_path = sys.argv[2]
    
    generate_embeddings(image_dir, db_path)