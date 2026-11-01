
import os
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

src = r"D:\Personal-AI\photos\heic_backup"
dst = r"D:\Personal-AI\photos\reference"

os.makedirs(dst, exist_ok=True)

files = [
    f for f in os.listdir(src)
    if f.lower().endswith((".heic", ".heif"))
]

count = 0

for f in files:
    output = os.path.join(
        dst, os.path.splitext(f)[0] + ".jpg"
    )

    if os.path.exists(output):
        print(f"Skipped (already exists): {f}")
        continue

    with Image.open(os.path.join(src, f)) as img:
        img.convert("RGB").save(
            output, "JPEG", quality=95
        )

    count += 1
    print(f"Converted: {f}")

print(f"\nComplete! {count} photos converted.")
