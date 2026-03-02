# Registers students by processing their photos and creating embeddings
# Run: python scripts/register_students.py

import os
import pickle
import numpy as np
import sys
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.face_engine import FaceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STUDENT_PHOTOS_DIR = "data/student_photos"
EMBEDDINGS_FILE = "data/embeddings.pkl"
MIN_PHOTOS_WARN = 3
MIN_CONFIDENCE = 0.7


def load_existing(filepath: str) -> dict:
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            existing = pickle.load(f)
        logger.info(f"Loaded existing database: {len(existing)} students")
        return existing
    logger.info("No existing database found — starting fresh")
    return {}


def get_image_files(folder: str) -> list:
    valid_ext = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1] in valid_ext]


def register_one_student(student_name: str, student_folder: str, engine: FaceEngine) -> np.ndarray | None:
    image_files = get_image_files(student_folder)

    if not image_files:
        print(f"  ✗ No images found in folder")
        return None

    if len(image_files) < MIN_PHOTOS_WARN:
        print(f"  ⚠ Only {len(image_files)} photo(s) — recommend {MIN_PHOTOS_WARN}+ for accuracy")

    embeddings_collected, skipped = [], []

    for photo_file in sorted(image_files):
        photo_path = os.path.join(student_folder, photo_file)
        faces = engine.get_embeddings(photo_path)

        if len(faces) == 0:
            skipped.append(f"{photo_file} (no face)")
            continue
        if len(faces) > 1:
            skipped.append(f"{photo_file} (multiple faces)")
            continue

        face = faces[0]
        if face["confidence"] < MIN_CONFIDENCE:
            skipped.append(f"{photo_file} (low confidence: {face['confidence']:.2f})")
            continue

        embeddings_collected.append(face["embedding"])
        print(f"  ✓ {photo_file:30s} conf: {face['confidence']:.3f}")

    for s in skipped:
        print(f"  ✗ {s}")

    if not embeddings_collected:
        print(f"  ✗ FAILED — no valid photos processed")
        return None

    avg = np.mean(embeddings_collected, axis=0)
    avg = avg / np.linalg.norm(avg)

    print(f"  ✅ Registered using {len(embeddings_collected)}/{len(image_files)} photos")
    return avg


def register_all(engine: FaceEngine) -> dict:
    if not os.path.exists(STUDENT_PHOTOS_DIR):
        logger.error(f"Student photos folder not found: {STUDENT_PHOTOS_DIR}")
        return {}

    all_embeddings = load_existing(EMBEDDINGS_FILE)
    student_folders = sorted([f for f in os.listdir(STUDENT_PHOTOS_DIR)
                             if os.path.isdir(os.path.join(STUDENT_PHOTOS_DIR, f))])

    if not student_folders:
        logger.error("No student folders found inside student_photos/")
        return {}

    print(f"\nFound {len(student_folders)} student folders\n{'─'*50}")

    success, failed = [], []

    for student_name in student_folders:
        student_folder = os.path.join(STUDENT_PHOTOS_DIR, student_name)
        print(f"\n[{student_name}]")

        embedding = register_one_student(student_name, student_folder, engine)

        if embedding is not None:
            all_embeddings[student_name] = embedding
            success.append(student_name)
        else:
            failed.append(student_name)

    print(f"\n{'='*50}")
    print(f"REGISTRATION COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")
    print(f"  ✅ Successfully registered : {len(success)}")
    print(f"  ✗  Failed                 : {len(failed)}")

    if failed:
        print(f"\n  Failed students:")
        for name in failed:
            print(f"    - {name}")

    print(f"\n  Registered students:")
    for name in sorted(all_embeddings.keys()):
        print(f"    ✓ {name}")
    print(f"{'='*50}")

    return all_embeddings


def save(embeddings: dict):
    os.makedirs("data", exist_ok=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info(f"Saved to {EMBEDDINGS_FILE}")


def verify(embeddings: dict):
    with open(EMBEDDINGS_FILE, "rb") as f:
        loaded = pickle.load(f)

    print(f"\nVerification:")
    print(f"  Students in file : {len(loaded)}")
    print(f"  Embedding shape  : {next(iter(loaded.values())).shape}")
    all_correct = all(v.shape == (512,) for v in loaded.values())
    print(f"  All embeddings   : {'✅ correct shape' if all_correct else '⚠ wrong shape'}")


if __name__ == "__main__":
    engine = FaceEngine(det_size=(640, 640), min_confidence=0.6)
    embeddings = register_all(engine)

    if embeddings:
        save(embeddings)
        verify(embeddings)
    else:
        print("\nNothing saved. Fix errors above first.")