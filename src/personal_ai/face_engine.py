from insightface.app import FaceAnalysis

from .config import (
    MODEL_NAME,
    EXECUTION_PROVIDERS,
    DETECTION_SIZE,
)


class FaceEngine:
    """Initializes and manages the InsightFace engine."""

    def __init__(self):
        self.providers = list(EXECUTION_PROVIDERS)

        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=self.providers,
        )

        ctx_id = (
            0 if "CUDAExecutionProvider" in self.providers else -1
        )

        self.app.prepare(
            ctx_id=ctx_id,
            det_size=DETECTION_SIZE,
        )

    def detect_faces(self, image):
        """Detect faces in an OpenCV image."""
        if image is None:
            raise ValueError("Input image cannot be empty.")

        return self.app.get(image)

    def get_embedding(self, face):
        """Extract a normalized face embedding."""
        embedding = face.normed_embedding

        if embedding is None:
            raise ValueError("Could not extract face embedding.")

        return embedding

    def get_provider_info(self):
        return self.providers