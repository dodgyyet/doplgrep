# doplgrep

`doplgrep` is a novelty CLI tool for face similarity search. It lets you build a local face embedding database and query it with a new image to find nearest matches.

The project is designed for experimentation and local demos, not identity verification.

## What It Does

- Builds embeddings from an image directory (`--mkdb`)
- Builds an HNSW index for faster search on larger datasets (`--mkidx`)
- Queries similar faces from a local SQLite database
- Optionally opens matched images from the CLI

## Setup

1. Create and activate a virtual environment.
2. Install the package in editable mode.
3. Authenticate with Hugging Face and request access to the DINOv3 model.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
huggingface-cli login
```

## Data Preparation

Use your own image collection for normal usage.

Example dataset source for demos:
- Mugshot dataset download: [https://huggingface.co/datasets/bitmind/idoc-mugshots-images](https://huggingface.co/datasets/bitmind/idoc-mugshots-images)

To create a local embedding database from an image directory:

```bash
doplgrep --mkdb data/faces/ embeddings/faces.sqlite
```

For larger datasets, build the HNSW index:

```bash
doplgrep --mkidx embeddings/faces.sqlite
```

## Brief Use Guide

1. Prepare a query image with one clear face.
2. Run a query against your database.
3. Review top matches in the terminal.

```bash
doplgrep query.jpg embeddings/faces.sqlite --top 5 -v
```

Optional: open returned matches automatically.

```bash
doplgrep query.jpg embeddings/faces.sqlite --top 3 --open
```

## Notes

- Best results come from front-facing, unblurred images with a single face.
- This tool is for creative and technical exploration.
