from datasets import load_dataset
import os
import random
import time
from PIL import Image

# Number of images to download for demo
NUM_IMAGES = 5000  # Set None for full dataset (~130k) if you want to thug out 1-3 hours

# Directory to save images
SAVE_DIR = "mugshots"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load dataset (train split)
ds = load_dataset("bitmind/idoc-mugshots-images", split="train")

# Shuffle indices and take NUM_IMAGES
indices = list(range(len(ds)))
random.shuffle(indices)
if NUM_IMAGES:
    indices = indices[:NUM_IMAGES]

total_images = len(indices)

batch_times = []  # track times for each 100-image batch
start_time = time.time()  # overall timer

for i, idx in enumerate(indices):
    # The dataset already returns PIL images
    img = ds[idx]['image']
    # Save as PNG without extra metadata
    img.save(os.path.join(SAVE_DIR, f"{i:05d}.png"), pnginfo=None)

    # Every 100 images, compute ETA
    if (i + 1) % 100 == 0 or (i + 1) == total_images:
        batch_end = time.time()
        batch_elapsed = batch_end - start_time if i < 100 else batch_elapsed  # first batch fallback
        batch_times.append(batch_elapsed if i < 100 else batch_elapsed)
        start_time = batch_end

        avg_time = sum(batch_times) / len(batch_times)
        remaining_batches = (total_images - (i + 1)) / 100
        eta_sec = avg_time * remaining_batches

        eta_min, eta_sec = divmod(eta_sec, 60)
        progress = (i + 1) / total_images * 100
        print(f"{i+1}/{total_images} images ({progress:.1f}%) - ETA: {int(eta_min)}m {int(eta_sec)}s")

print("Done! Images saved to:", SAVE_DIR)