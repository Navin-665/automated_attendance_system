# Evaluates attendance system accuracy by comparing against ground truth
# Usage: python scripts/evaluate.py

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.attendance import AttendanceProcessor


def calculate_metrics(TP, TN, FP, FN):
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    far = FP / (FP + TN) if (FP + TN) > 0 else 0
    frr = FN / (FN + TP) if (FN + TP) > 0 else 0
    return accuracy, precision, recall, f1, far, frr


def evaluate(photo_path: str, ground_truth: dict, threshold: float = 0.4):
    processor = AttendanceProcessor(threshold=threshold)
    result = processor.process(photo_path)

    if not result:
        print("ERROR: Could not process photo")
        return

    TP = TN = FP = FN = 0
    details = []

    for row in result["attendance"]:
        student = row["student"]
        system_status = row["status"]
        actual_status = ground_truth.get(student, "ABSENT")

        if actual_status == "PRESENT" and system_status == "PRESENT":
            outcome, TP = "TP", TP + 1
        elif actual_status == "ABSENT" and system_status == "ABSENT":
            outcome, TN = "TN", TN + 1
        elif actual_status == "ABSENT" and system_status == "PRESENT":
            outcome, FP = "FP ← WRONG (proxy risk)", FP + 1
        else:
            outcome, FN = "FN ← WRONG (missed student)", FN + 1

        details.append({"student": student, "actual": actual_status, "system": system_status, 
                       "outcome": outcome, "score": row["score"]})

    accuracy, precision, recall, f1, far, frr = calculate_metrics(TP, TN, FP, FN)

    print(f"\n{'='*60}")
    print(f"  EVALUATION REPORT")
    print(f"  Photo     : {photo_path}")
    print(f"  Threshold : {threshold}")
    print(f"{'='*60}")
    print(f"\nPer Student Breakdown:")
    print(f"{'─'*60}")
    print(f"  {'Student':35s} {'Actual':8s} {'System':8s} Outcome")
    print(f"{'─'*60}")
    for d in details:
        print(f"  {d['student']:35s} {d['actual']:8s} {d['system']:8s} {d['outcome']}")

    print(f"\n{'─'*60}")
    print(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
    print(f"{'─'*60}")
    print(f"\nMetrics:")
    print(f"  Accuracy  : {accuracy * 100:.1f}%")
    print(f"  Precision : {precision * 100:.1f}%")
    print(f"  Recall    : {recall * 100:.1f}%")
    print(f"  F1 Score  : {f1 * 100:.1f}%")
    print(f"\nBiometric Metrics:")
    print(f"  FAR       : {far * 100:.1f}%  (false acceptance)")
    print(f"  FRR       : {frr * 100:.1f}%  (false rejection)")
    print(f"\n{'='*60}")
    print(f"\nInterpretation:")
    print(f"  {'✅' if accuracy >= 0.85 else '⚠'} Accuracy {accuracy*100:.0f}% — {'meets' if accuracy >= 0.85 else 'below'} target (>85%)")
    print(f"  {'✅' if far == 0 else '⚠'} FAR {far*100:.0f}% — {'no' if far == 0 else FP} proxy attendance")
    print(f"  {'✅' if frr <= 0.1 else '⚠'} FRR {frr*100:.0f}% — {FN if frr > 0.1 else 'minimal'} missed students")
    print(f"{'='*60}\n")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "FAR": far, "FRR": frr, "TP": TP, "TN": TN, "FP": FP, "FN": FN}


if __name__ == "__main__":
    ground_truth = {
        "student_1_naveen": "PRESENT",
        "student_2_jeevan": "PRESENT",
        "student_3_akshay": "PRESENT",
        "student_4_sayam": "PRESENT",
        "student_5_lohith": "PRESENT",
        "student_6_razzaq": "PRESENT",
        "student_7_sidda": "ABSENT",
    }

    metrics = evaluate(
        photo_path="data/class_photos/group_image.jpg",
        ground_truth=ground_truth,
        threshold=0.4
    )