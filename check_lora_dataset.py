
from pathlib import Path
from PIL import Image

folder = Path(r"D:\Personal-AI\photos\lora_training")

extensions = {".jpg", ".jpeg", ".png", ".webp"}

print("\n=== LoRA Dataset Inspection ===\n")

total = 0

for file in sorted(folder.iterdir()):
    if file.suffix.lower() not in extensions:
        continue

    try:
        with Image.open(file) as img:
            width, height = img.size
            print(f"{file.name:35} {width:5} x {height:5}")
            total += 1
    except Exception as e:
        print(f"ERROR: {file.name} - {e}")

print(f"\nTotal valid images: {total}")
