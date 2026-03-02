# Command-line script to process class photos and generate attendance
# Usage: python scripts/process_class_photo.py --photo <path> --threshold <value>
# Or: python scripts/process_class_photo.py --photos <path1> <path2> ...

import os
import sys
import cv2
import argparse
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.face_engine import FaceEngine
from modules.matcher import Matcher
from modules.attendance import AttendanceProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def draw_results(class_photo_path: str, output_path: str, threshold: float = 0.4):
    img = cv2.imread(class_photo_path)
    if img is None:
        return
    
    # Use recommended detection settings for classroom photos
    engine = FaceEngine(det_size=(1280, 1280), min_confidence=0.5)
    matcher = Matcher(threshold=threshold)
    faces = engine.get_embeddings(class_photo_path)

    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        match = matcher.match(face["embedding"])
        color = (0, 200, 0) if match["matched"] else (0, 0, 220)
        label = match["name"].split("_")[-1].upper() if match["matched"] else "Unknown"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if match["matched"]:
            cv2.putText(img, f"{match['score']:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.imwrite(output_path, img)
    logger.info(f"Visual result saved: {output_path}")


def process_multiple(photo_paths: list, threshold: float = 0.4) -> dict:
    """
    Process multiple photos and merge results.
    Anyone found in ANY photo is marked PRESENT.
    """
    processor = AttendanceProcessor(threshold=threshold)
    all_merged_results = None
    present_students = set()

    for photo in photo_paths:
        if not os.path.exists(photo):
            logger.warning(f"Photo not found: {photo}")
            continue
            
        result = processor.process(photo)
        if not result:
            continue
            
        if all_merged_results is None:
            # Initialize with first result structure
            all_merged_results = result
        
        # Track who is present in ANY photo
        for row in result["attendance"]:
            if row["status"] == "PRESENT":
                present_students.add(row["student"])

    if all_merged_results:
        # Update merged results based on combined data
        for row in all_merged_results["attendance"]:
            if row["student"] in present_students:
                row["status"] = "PRESENT"
            else:
                row["status"] = "ABSENT"
                row["score"] = 0.0
        
        # Recalculate counts
        all_merged_results["present_count"] = len(present_students)
        all_merged_results["absent_count"] = all_merged_results["total"] - len(present_students)
        
    return all_merged_results


def run(photo_paths: list, threshold: float = 0.4):
    if not photo_paths:
        print("ERROR: No photos provided.")
        return

    if not os.path.exists("data/embeddings.pkl"):
        print("ERROR: No student database found. Run register_students.py first.")
        return

    print(f"\nProcessing {len(photo_paths)} photos...")
    print(f"Threshold : {threshold}\n{'─'*50}")

    if len(photo_paths) > 1:
        result = process_multiple(photo_paths, threshold)
    else:
        processor = AttendanceProcessor(threshold=threshold)
        result = processor.process(photo_paths[0])

    if not result:
        print("ERROR: Processing failed.")
        return

    AttendanceProcessor(threshold=threshold).print_summary(result)
    csv_path = AttendanceProcessor(threshold=threshold).save_csv(result)

    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate annotated images for each photo
    for i, photo_path in enumerate(photo_paths):
        visual_path = f"output/annotated_{timestamp}_{i+1}.jpg"
        draw_results(photo_path, visual_path, threshold)
        print(f"  Annotated photo {i+1}: {visual_path}")

    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*50}")
    print(f"  CSV saved      : {csv_path}")
    print(f"  Present        : {result['present_count']}/{result['total']}")
    print(f"  Absent         : {result['absent_count']}/{result['total']}")
    print(f"  Unknown faces  : {result['unknown_faces']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process class photo for attendance")
    parser.add_argument("--photo", type=str, help="Path to single class photo")
    parser.add_argument("--photos", type=str, nargs="+", help="Paths to multiple class photos")
    parser.add_argument("--threshold", type=float, default=0.4, help="Match threshold (0.3-0.6)")
    args = parser.parse_args()
    
    target_photos = []
    if args.photos:
        target_photos = args.photos
    elif args.photo:
        target_photos = [args.photo]
    else:
        # Default fallback
        target_photos = ["data/class_photos/group_image.jpg"]
        
    run(target_photos, args.threshold)