
import os
import cv2
from insightface.app import FaceAnalysis

folder = r"D:\Personal-AI\photos\reference"

print("Loading InsightFace model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

print("\nScanning reference images...\n")

total = 0
detected = 0
no_face = []
failed = []

for filename in os.listdir(folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    total += 1
    path = os.path.join(folder, filename)

    img = cv2.imread(path)

    if img is None:
        failed.append(filename)
        print(f"[ERROR] Cannot read: {filename}")
        continue

    faces = app.get(img)

    if len(faces) > 0:
        detected += 1
        print(f"[OK] {filename} - {len(faces)} face(s)")
    else:
        no_face.append(filename)
        print(f"[NO FACE] {filename}")

print("\n========== RESULTS ==========")
print(f"Total images: {total}")
print(f"Faces detected: {detected}")
print(f"No face detected: {len(no_face)}")
print(f"Unreadable images: {len(failed)}")

if no_face:
    print("\nImages with no face:")
    for f in no_face:
        print(f)

if failed:
    print("\nUnreadable images:")
    for f in failed:
        print(f)
