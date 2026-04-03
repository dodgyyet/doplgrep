# Architecture Overview

## Project Scope

- Name: `doplgrep`
- Type: Local novelty CLI for face similarity matching
- Language: Python
- Runtime target: macOS (Apple Silicon) and Linux
- Non-goals: model training, identity verification, cloud deployment, GUI

## Core Stack

- Embedding model: DINOv3 ViT-L (`facebook/dinov3-vitl16-pretrain-lvd1689m`)
- Face detection: MediaPipe Face Detector
- Vector storage: SQLite (`face_embeddings` table)
- Fast retrieval: HNSW (`hnswlib` index file)
- Similarity metric: cosine

## Pipeline

### Build Database (`--mkdb`)
1. Recursively load images from an input directory.
2. Detect and crop the primary face.
3. Generate a 1024-d embedding.
4. Normalize embedding.
5. Store image path + embedding bytes in SQLite.

### Build Index (`--mkidx`)
1. Read embeddings from SQLite.
2. Build HNSW graph.
3. Save `.hnsw` index next to the SQLite database.

### Query
1. Load and crop query image.
2. Generate and normalize query embedding.
3. Search via HNSW when index exists.
4. Fallback to brute-force cosine search when no index exists.
5. Return top-N matches.

## CLI Surface

```bash
doplgrep --mkdb <image_dir> <output_db>
doplgrep --mkidx <database.sqlite>
doplgrep <query_image> <database.sqlite> --top <N> [-v] [--open]
```

## Dataset Notes

This project is dataset agnostic. Any image collection can be used as a local face database.

Example source used during development:
- [https://huggingface.co/datasets/bitmind/idoc-mugshots-images](https://huggingface.co/datasets/bitmind/idoc-mugshots-images)
