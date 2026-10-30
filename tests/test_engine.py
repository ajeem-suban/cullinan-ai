import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from personal_ai.face_recognition import (
    cosine_similarity,
    is_same_person,
)


class TestFaceRecognition(unittest.TestCase):

    def test_identical_embeddings(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])

        self.assertAlmostEqual(
            cosine_similarity(a, b),
            1.0,
        )

    def test_opposite_embeddings(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])

        self.assertAlmostEqual(
            cosine_similarity(a, b),
            -1.0,
        )

    def test_matching_threshold(self):
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])

        self.assertTrue(
            is_same_person(a, b, threshold=0.8)
        )

    def test_different_embeddings(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])

        self.assertFalse(
            is_same_person(a, b, threshold=0.8)
        )

    def test_dimension_mismatch(self):
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])

        with self.assertRaises(ValueError):
            cosine_similarity(a, b)

    def test_zero_embedding(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])

        with self.assertRaises(ValueError):
            cosine_similarity(a, b)


if __name__ == "__main__":
    unittest.main()