from pathlib import Path

from personal_ai.face_engine import FaceEngine
from personal_ai.face_detection import process_directory


ROOT = Path(__file__).parent

PHOTO_DIR = ROOT / "photos" / "reference"
OUTPUT_DIR = ROOT / "output"


def main():
    print("Initializing PERSONAL-AI Face Engine...")

    engine = FaceEngine()

    print("Face engine initialized.")
    print(f"Execution providers: {engine.get_provider_info()}")

    process_directory(
        engine=engine,
        input_dir=PHOTO_DIR,
        output_dir=OUTPUT_DIR,
    )

    print("PERSONAL-AI face detection completed.")


if __name__ == "__main__":
    main()