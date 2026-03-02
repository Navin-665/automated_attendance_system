# Matches face embeddings against registered students using cosine similarity
# Uses L2-normalized embeddings for fast and accurate matching

import pickle
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)
EMBEDDINGS_FILE = "data/embeddings.pkl"


class Matcher:
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.stored = {}
        self._load()

    def _load(self):
        if not os.path.exists(EMBEDDINGS_FILE):
            logger.error(f"No embeddings file found at {EMBEDDINGS_FILE}")
            return
        with open(EMBEDDINGS_FILE, "rb") as f:
            self.stored = pickle.load(f)
        logger.info(f"Loaded {len(self.stored)} student embeddings")

    def reload(self):
        self._load()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def match(self, unknown_embedding: np.ndarray) -> dict:
        if not self.stored:
            return {"name": "Unknown", "score": 0.0, "matched": False, "all_scores": {}}

        norm = np.linalg.norm(unknown_embedding)
        if norm == 0:
            return {"name": "Unknown", "score": 0.0, "matched": False, "all_scores": {}}
        unknown_embedding = unknown_embedding / norm

        all_scores = {name: round(self._cosine_similarity(unknown_embedding, emb), 4)
                     for name, emb in self.stored.items()}
        
        best_name = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_name]

        return {
            "name": best_name if best_score >= self.threshold else "Unknown",
            "score": best_score,
            "matched": best_score >= self.threshold,
            "all_scores": all_scores
        }

    def match_all(self, embeddings: list) -> list:
        results = []
        for face in embeddings:
            result = self.match(face["embedding"])
            result["bbox"] = face["bbox"]
            result["confidence"] = face["confidence"]
            results.append(result)
        return results