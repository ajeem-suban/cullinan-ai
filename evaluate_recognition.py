
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Paths
REFERENCE_FILE = r"D:\Personal-AI\output\face_embeddings.npz"
GENUINE_DIR = r"D:\Personal-AI\photos\evaluation\genuine"
IMPOSTOR_DIR = r"D:\Personal-AI\photos\evaluation\impostor"

THRESHOLD = 0.45  # Provisional; we will calibrate it

# Load reference embeddings
data = np.load(REFERENCE_FILE)
reference_embeddings = data["embeddings"]

# Initialize InsightFace
print("Loading InsightFace model...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def evaluate_folder(folder, expected_match):
    results = []

    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return results

    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"[ERROR] Cannot read: {filename}")
            continue

        faces = app.get(img)

        if len(faces) != 1:
            print(f"[SKIP] {filename}: detected {len(faces)} faces")
            continue

        embedding = faces[0].normed_embedding
        scores = np.dot(reference_embeddings, embedding)
        best_score = float(np.max(scores))

        predicted_match = best_score >= THRESHOLD
        correct = predicted_match == expected_match

        results.append((best_score, correct, filename))

        status = "MATCH" if predicted_match else "UNKNOWN"
        print(f"{filename}: {best_score:.4f} -> {status}")

    return results


print("\n========== GENUINE TESTS ==========")
genuine = evaluate_folder(GENUINE_DIR, True)

print("\n========== IMPOSTOR TESTS ==========")
impostor = evaluate_folder(IMPOSTOR_DIR, False)

# Calculate metrics
total_genuine = len(genuine)
total_impostor = len(impostor)

true_accepts = sum(score >= THRESHOLD for score, _, _ in genuine)
false_rejects = total_genuine - true_accepts

false_accepts = sum(score >= THRESHOLD for score, _, _ in impostor)
true_rejects = total_impostor - false_accepts

print("\n========== EVALUATION RESULTS ==========")
print(f"Threshold: {THRESHOLD:.2f}")
print(f"Genuine samples: {total_genuine}")
print(f"Impostor samples: {total_impostor}")

if total_genuine:
    print(f"Genuine Acceptance Rate (GAR): {true_accepts / total_genuine:.2%}")
    print(f"False Rejection Rate (FRR): {false_rejects / total_genuine:.2%}")

if total_impostor:
    print(f"False Acceptance Rate (FAR): {false_accepts / total_impostor:.2%}")
    print(f"True Rejection Rate: {true_rejects / total_impostor:.2%}")

print("\nEvaluation complete.")
