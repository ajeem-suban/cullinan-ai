MODEL_NAME = "buffalo_l"

# Keep CPU for now. We can evaluate CUDA later.
EXECUTION_PROVIDERS = ("CPUExecutionProvider",)

DETECTION_SIZE = (640, 640)

SUPPORTED_IMAGE_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}