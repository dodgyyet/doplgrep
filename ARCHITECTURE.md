Name: Doplgrep CLI Tool (Min-Max Demo)
Platform: macOS (Apple Silicon)
Disk: ~2 TB available
Demo dataset: ipoc_mugshots
Input constraint: single, unblurred faces only, not AI generated, not celebs, clean background
Model: DINOv3 ViT-L
Inference: PyTorch MPS, bf16
Embedding: 1024-dimensional
Vector DB: SQLite + sqlite-vec 
Distance metric: cosine
Algorithm: HNSW 
Language: Python
Pipeline:

Today (fast demo, min-max):
	1.	Stream → Crop → Embed → Query DB → Display results

Later (full dataset):
	•	Store → Build HNSW → Delete (for full dataset)

Non-goals: training, full dataset download, identity verification, cloud, FAISS, GUI
⸻

CLI Responsibilities (Min-Max Demo Today)
	1.	Load DINOv3 ViT-L model on MPS (bf16).
	2.	Preprocess input image (crop + resize to 224×224).
	3.	Embed face into 1024-dimensional vector.
	4.	Query prebuilt SQLite vector DB (sqlite-vec) using cosine + HNSW.
	5.	Return top-N matches to console (paths + cosine distance)
	6.	Open matched image(s) automatically (optional for demo).

doplgrep path/to/input.jpg path/to/demo_database.db [--top N] [-v]

Flags:
	--mkdb = make database from inputs images directory into output directory
	--mkidx = make HNSW index to speed up large similarity searches
	--top N = number of matches to return (default 1)
	-v → verbose output (paths + distance scores)


Step-by-Step Workflow

Step 1 – Prepare Environment:
	Install Python & required libraries: pip install -e .
	Log in with huggingface and request access to Dinov3

Step 2 – Prepare Demo Database:
	•	Demo uses 130k images -> precompute embeddings -> HNSW index
	•	Store read-only SQLite vector DB for CLI testing.

Step 3 – Run CLI:
	Run doplgrep
	Run doplgrep input.jpg demo_database.db
	CLI loads model, embeds input, queries prebuilt DB, returns top matches.
	•	Fast feedback; works entirely today.

Step 4 – Optional: expand dataset later (hours/days):
	•	Download larger FFHQ or LAION-Face subset
	•	Recompute embeddings, rebuild HNSW, update SQLite DB