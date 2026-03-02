# Processes class photos to generate attendance records
# Connects FaceEngine (detection) + Matcher (identification)

import os
import sys
import pickle
import csv
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.face_engine import FaceEngine
from modules.matcher import Matcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDINGS_FILE = "data/embeddings.pkl"
OUTPUT_DIR = "output"


class AttendanceProcessor:
    def __init__(self, threshold=0.4):
        self.engine = FaceEngine(det_size=(640, 640), min_confidence=0.6)
        self.matcher = Matcher(threshold=threshold)
        self.all_students = self._load_students()

    def _load_students(self) -> list:
        if not os.path.exists(EMBEDDINGS_FILE):
            logger.error("embeddings.pkl not found. Run register_students.py first.")
            return []
        with open(EMBEDDINGS_FILE, "rb") as f:
            stored = pickle.load(f)
        logger.info(f"Loaded {len(stored)} registered students")
        return list(stored.keys())

    def process(self, class_photo_path: str) -> dict:
        if not self.all_students or not os.path.exists(class_photo_path):
            logger.error("Cannot process: missing students or photo")
            return {}

        logger.info(f"Processing: {class_photo_path}")
        faces = self.engine.get_embeddings(class_photo_path)
        logger.info(f"Faces detected: {len(faces)}")

        present, unknown = {}, 0
        for i, face in enumerate(faces):
            result = self.matcher.match(face["embedding"])
            if result["matched"]:
                name, score = result["name"], result["score"]
                if name not in present or score > present[name]:
                    present[name] = score
                logger.info(f"  Face {i+1} → {name} ({score:.3f})")
            else:
                unknown += 1
                logger.info(f"  Face {i+1} → Unknown ({result['score']:.3f})")

        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M:%S")

        attendance = [{"student": s, "status": "PRESENT" if s in present else "ABSENT",
                      "score": round(present.get(s, 0.0), 4), "date": date_str, "time": time_str}
                     for s in sorted(self.all_students)]

        return {
            "photo": class_photo_path,
            "date": date_str,
            "time": time_str,
            "total": len(self.all_students),
            "present_count": sum(1 for a in attendance if a["status"] == "PRESENT"),
            "absent_count": sum(1 for a in attendance if a["status"] == "ABSENT"),
            "faces_detected": len(faces),
            "unknown_faces": unknown,
            "attendance": attendance
        }

    def save_csv(self, result: dict) -> str:
        if not result:
            return ""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(OUTPUT_DIR, f"attendance_{timestamp}.csv")
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "time", "student", "status", "score"])
            writer.writeheader()
            writer.writerows(result["attendance"])
        logger.info(f"Attendance saved: {filepath}")
        return filepath

    def print_summary(self, result: dict):
        if not result:
            return
        print(f"\n{'='*55}")
        print(f"  ATTENDANCE REPORT")
        print(f"{'='*55}")
        print(f"  Photo    : {result['photo']}")
        print(f"  Date     : {result['date']}  {result['time']}")
        print(f"  Detected : {result['faces_detected']} faces ({result['unknown_faces']} unknown)")
        print(f"  Present  : {result['present_count']}/{result['total']}")
        print(f"  Absent   : {result['absent_count']}/{result['total']}")
        print(f"{'─'*55}")
        for row in result["attendance"]:
            symbol = "✅" if row["status"] == "PRESENT" else "❌"
            score = f"({row['score']:.3f})" if row["status"] == "PRESENT" else ""
            print(f"  {symbol}  {row['student']:35s} {row['status']} {score}")
        print(f"{'='*55}\n")