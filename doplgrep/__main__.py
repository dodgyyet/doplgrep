"""CLI entry point for doplgrep."""
import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
from .utils import open_file, open_image, FaceDetector
from .image_embedder import ImageEmbedder
from .query_doppelganger import query_database_hnsw
from .generate_embeddings import bug_squasher


def main():
    """Main CLI entry point with subcommand routing."""
    parser = argparse.ArgumentParser(
        description="doplgrep - Customizable face doppelganger search CLI tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create database from images
  doplgrep --mkdb data/faces/ embeddings/faces.sqlite
  
  # Build HNSW index for fast search (optional, for 10K+ images)
  doplgrep --mkidx embeddings/faces.sqlite
  
  # Query for similar faces
  doplgrep query.jpg embeddings/faces.sqlite --top 5 -v --open
        """
    )
    
    parser.add_argument(
        "--mkdb",
        action="store_true",
        help="Generate embeddings database from image directory"
    )
    parser.add_argument(
        "--mkidx",
        action="store_true",
        help="Build HNSW index from existing database (recommended for 10K+ images)"
    )
    parser.add_argument(
        "input",
        help="Input image (query mode) OR image directory (--mkdb) OR database path (--mkidx)"
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Database path (required for query and --mkdb modes)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=1,
        help="Number of top matches to return"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with full paths and similarity scores"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open top N matched images"
    )
    
    args = parser.parse_args()
    
    # Route to appropriate function
    if args.mkdb:
        if not args.output:
            parser.error("--mkdb requires both <image_dir> and <output_db>")
        cmd_mkdb(args.input, args.output)
    
    elif args.mkidx:
        cmd_mkidx(args.input)
    
    else:
        if not args.output:
            parser.error("Query mode requires both <input_image> and <database>")
        cmd_query(args)


def cmd_mkdb(image_dir: str, db_path: str):
    """Generate embeddings database from image directory."""
    from .generate_embeddings import generate_embeddings
    
    print(f"Creating database from: {image_dir}")
    print(f"Output database: {db_path}\n")
    
    bug_squasher(image_dir, "cmd_mkdb - image_dir")
    bug_squasher(db_path, "cmd_mkdb - db_path")
    generate_embeddings(image_dir, db_path)
    
    # print(f"\nDatabase created successfully!")
    # print(f"doplgrep --mkidx {db_path}")


def cmd_mkidx(db_path: str):
    """Build HNSW index from existing database."""
    from .vector_index import HNSWIndex
    
    index_path = db_path.replace('.sqlite', '.hnsw')
    
    print(f"Building HNSW index for: {db_path}")
    print(f"Output index: {index_path}\n")
    
    index = HNSWIndex()
    index.build_from_database(db_path)
    index.save(index_path)
    
    print(f"\nIndex built successfully!")
    print(f"Queries will now use fast HNSW search")


def cmd_query(args):
    """Query database for similar faces."""
 
    
    # Initialize embedder
    print("Loading DINOv3 model...")
    embedder = ImageEmbedder()
    
    # Embed query image
    print(f"Embedding query image: {args.input}")    
    cropped_img = FaceDetector().detect_and_crop(args.input)
    query_embedding = embedder.embed(cropped_img)
    
    # Normalize
    query_embedding_np = query_embedding.cpu().numpy().flatten()
    query_embedding_np = query_embedding_np / np.linalg.norm(query_embedding_np)
    
    # Query database
    print(f"Searching database: {args.output}\n")
    results = query_database_hnsw(args.output, query_embedding_np, args.top)
    
    # Display results
    print(f"{'='*60}")
    print(f"Top {args.top} matches:")
    print(f"{'='*60}\n")
    
    for i, (similarity, img_path) in enumerate(results, 1):
        if args.verbose:
            print(f"{i}. {img_path}")
            print(f"Cosine similarity: {similarity:.6f}\n")
        else:
            print(f"{i}. {Path(img_path).name:<40} (similarity: {similarity:.4f})")
        
        # Optional: open images
        if args.open:
            open_file(img_path)
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()