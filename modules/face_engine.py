# Face detection and embedding generation using InsightFace
# Wraps RetinaFace (detection) + ArcFace (512-D embeddings)

import cv2
import numpy as np
import logging
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


class FaceEngine:
    def __init__(self, det_size=(1280, 1280), min_confidence=0.5):
        self.det_size = det_size
        self.min_confidence = min_confidence
        self._load_model()

    def _load_model(self):
        logger.info("Loading InsightFace model (buffalo_l)...")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=self.det_size)
        logger.info("Model ready.")

    def _enhance_image(self, img):
        """
        Improve image quality before detection.
        Helps with poor classroom lighting.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel (improves contrast)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def get_embeddings(self, image_path: str) -> list:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return []

        img = self._enhance_image(img)

        raw_faces = self.app.get(img)
        if not raw_faces:
            logger.warning(f"No faces found in: {image_path}")
            return []

        results = []
        for face in raw_faces:
            confidence = float(face.det_score)
            if confidence < self.min_confidence:
                continue

            x1, y1, x2, y2 = face.bbox.astype(int)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            face_crop = img[y1:y2, x1:x2]

            if face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                continue

            results.append({
                "embedding": face.embedding,
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "face_crop": face_crop
            })

        logger.info(f"Image: {image_path} | Detected: {len(raw_faces)} | Valid: {len(results)}")
        return results

    def draw_boxes(self, image_path: str, faces: list, output_path: str):
        img = cv2.imread(image_path)
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"#{i+1} {face['confidence']:.2f}", (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(output_path, img)
        logger.info(f"Annotated image saved: {output_path}")