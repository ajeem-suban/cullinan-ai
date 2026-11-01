
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Paths
EMBEDDINGS_FILE = r"D:\Personal-AI\output\face_embeddings.npz"
TEST_DIR = r"D:\Personal-AI\photos\test"

# Recognition threshold (we will calibrate this)
THRESHOLD = 0.45

# Load reference embeddings
data = np.load(EMBEDDINGS_FILE)
reference_embeddings = data["embeddings"]
reference_filenames = data["filenames"]

print(f"Loaded {len(reference_embeddings)} reference embeddings.")

# Initialize InsightFace
print("Loading InsightFace...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# Process test images
for filename in os.listdir(TEST_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(TEST_DIR, filename)
    img = cv2.imread(path)

    if img is None:
        print(f"\nCannot read: {filename}")
        continue

    faces = app.get(img)

    print(f"\nTest image: {filename}")

    if len(faces) == 0:
        print("No face detected.")
        continue

    if len(faces) > 1:
        print(f"Warning: {len(faces)} faces detected.")

    for i, face in enumerate(faces):
        test_embedding = face.normed_embedding

        # Cosine similarity (normalized embeddings)
        similarities = np.dot(
            reference_embeddings, test_embedding
        )

        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        result = (
            "MATCH - Likely you"
            if best_score >= THRESHOLD
            else "UNKNOWN - No confident match"
        )

        print(f"Face {i + 1}: {result}")
        print(f"Similarity: {best_score:.4f}")
        print(f"Closest reference: {reference_filenames[best_index]}")

print("\nRecognition test complete.")
