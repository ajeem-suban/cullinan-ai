from pathlib import Path

import cv2

from .face_engine import FaceEngine
from .config import SUPPORTED_IMAGE_FORMATS


def load_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return image


def detect_image(engine, image_path):
    image = load_image(image_path)
    faces = engine.detect_faces(image)

    return image, faces


def save_annotated_image(engine, image, faces, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = engine.app.draw_on(image, faces)

    success = cv2.imwrite(str(output_path), result)

    if not success:
        raise IOError(f"Could not save image: {output_path}")

    return output_path


def process_directory(engine, input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    images = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_FORMATS
    ]

    if not images:
        print(f"No supported images found in {input_dir}")
        return

    for image_path in images:
        try:
            image, faces = detect_image(engine, image_path)

            print(
                f"{image_path.name}: "
                f"{len(faces)} face(s) detected"
            )

            if faces:
                destination = (
                    output_dir / f"{image_path.stem}_detected.jpg"
                )

                save_annotated_image(
                    engine,
                    image,
                    faces,
                    destination,
                )

                print(f"  Saved: {destination}")

        except Exception as error:
            print(f"Error processing {image_path.name}: {error}")