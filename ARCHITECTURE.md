Name: Doplgrep CLI Tool (Min-Max Demo)
Platform: macOS (Apple Silicon)
Disk: ~2 TB available
Dataset: FFHQ subset (~200–400k images for initial demo only)
Input constraint: single, unblurred faces only
Model: DINOv3 ViT-L
Inference: PyTorch MPS, bf16
Embedding: 1024-dimensional
Vector DB: SQLite + sqlite-vec (prebuilt demo DB, read-only)
Distance metric: cosine
Algorithm: HNSW (prebuilt index)
Language: Python
Pipeline:

Today (fast demo, min-max):
	1.	Stream → Crop → Embed → Query DB → Display results

Later (full dataset):
	•	Store → Build HNSW → Delete (for full dataset, optional for today)

Non-goals (today/demo): training, full dataset download, identity verification, cloud, FAISS, GUI

⸻

CLI Responsibilities (Min-Max Demo Today)
	1.	Load DINOv3 ViT-L model on MPS (bf16).
	2.	Preprocess input image (crop + resize to 224×224).
	3.	Embed face into 1024-dimensional vector.
	4.	Query prebuilt SQLite vector DB (sqlite-vec) using cosine + HNSW.
	5.	Return top-N matches to console (paths + cosine distance)
	6.	Open matched image(s) automatically (optional for demo).

doplgrep path/to/input.jpg path/to/demo_database.db [--top N] [-v]

Optional flags:
	•	--top N → number of matches to return (default 1)
	•	-v → verbose output (paths + distance scores)

Step-by-Step Fast Demo Workflow

Step 1 – Prepare Environment (minutes):
	•	Install Python & required libraries: torch, timm, Pillow, sqlite3, sqlite-vec.
	•	No full dataset download today.

Step 2 – Prepare Prebuilt Demo Database (minutes):
	•	Use a small subset (200–400k FFHQ images) and precompute embeddings + HNSW index.
	•	Store read-only SQLite vector DB for CLI testing.
	•	This DB is delivered with demo — no need to build today.

Step 3 – Run CLI (minutes):
	•	Run doplgrep input.jpg demo_database.db
	•	CLI loads model, embeds input, queries prebuilt DB, returns top matches.
	•	Fast feedback; works entirely today.

Step 4 – Optional: expand dataset later (hours/days):
	•	Download larger FFHQ or LAION-Face subset
	•	Recompute embeddings, rebuild HNSW, update SQLite DB