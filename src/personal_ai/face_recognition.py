import numpy as np


def cosine_similarity(embedding_a, embedding_b):
    """Calculate cosine similarity between two face embeddings."""

    a = np.asarray(embedding_a, dtype=np.float32).reshape(-1)
    b = np.asarray(embedding_b, dtype=np.float32).reshape(-1)

    if a.shape != b.shape:
        raise ValueError("Embedding dimensions must match.")

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Embeddings must contain finite values.")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Zero-length embeddings are invalid.")

    return float(np.dot(a, b) / (norm_a * norm_b))


def is_same_person(embedding_a, embedding_b, threshold):
    """
    Compare two face embeddings.

    Threshold must be calibrated using real validation images.
    """

    if not -1.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between -1 and 1.")

    similarity = cosine_similarity(
        embedding_a,
        embedding_b,
    )

    return similarity >= threshold