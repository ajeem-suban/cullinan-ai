
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Project paths
REFERENCE_DIR = r"D:\Personal-AI\photos\reference"
OUTPUT_DIR = r"D:\Personal-AI\output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "face_embeddings.npz")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize InsightFace
print("Loading InsightFace model...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

embeddings = []
filenames = []
skipped = []

print("\nGenerating face embeddings...\n")

for filename in os.listdir(REFERENCE_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(REFERENCE_DIR, filename)
    img = cv2.imread(path)

    if img is None:
        skipped.append(filename)
        print(f"[SKIP] Cannot read: {filename}")
        continue

    faces = app.get(img)

    # Skip images with zero or multiple faces
    if len(faces) != 1:
        skipped.append(filename)
        print(f"[SKIP] {filename}: {len(faces)} faces")
        continue

    # Extract and normalize embedding
    embedding = faces[0].normed_embedding
    embedding = embedding.astype(np.float32)

    embeddings.append(embedding)
    filenames.append(filename)

    print(f"[OK] {filename}")

# Save results
if embeddings:
    embeddings = np.array(embeddings, dtype=np.float32)

    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=embeddings,
        filenames=np.array(filenames)
    )

    print("\n========== RESULTS ==========")
    print(f"Successfully processed: {len(embeddings)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {OUTPUT_FILE}")
else:
    print("ERROR: No valid face embeddings generated.")
