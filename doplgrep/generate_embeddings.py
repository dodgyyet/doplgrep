"""Generate embeddings for a dataset and store in SQLite."""
import sqlite3
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .image_embedder import ImageEmbedder
from .utils import FaceDetector, open_image
from PIL import Image
import traceback
from itertools import chain

#Bug squasher - remove later 
def bug_squasher(x, where: str):
    if isinstance(x, Path):
        raise TypeError(f"[{where}] Expected str, got Path: {x}")
    if isinstance(x, Image.Image):
        raise TypeError(f"[{where}] Expected path or mp.Image, got PIL.Image")
    if not isinstance(x, str):
        raise TypeError(f"[{where}] Expected str, got {type(x)}")

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


    exts = ["png", "jpg", "jpeg", "heic", "webp"]
    image_paths = [str(p) for p in chain.from_iterable(Path(image_dir).rglob(f"*.{ext}") for ext in exts)]
    
    successful = 0
    for img_path in tqdm(image_paths):
        try:
            # Load and embed image
            bug_squasher(img_path, "generate_embeddings - img_path")
            detector = FaceDetector()
            cropped_img = detector.detect_and_crop(img_path)
            embedding = embedder.embed(cropped_img)
            
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
            bug_squasher(img_path, "generate_embeddings - error img_path")
            traceback.print_exc()
            break
    
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