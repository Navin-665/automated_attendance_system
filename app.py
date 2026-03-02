# Flask web interface for attendance system
# Run: python app.py | Open: http://localhost:5000

import os
import sys
import csv
import cv2
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from admin import admin as admin_blueprint

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.face_engine import FaceEngine
from modules.matcher import Matcher
from modules.attendance import AttendanceProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.register_blueprint(admin_blueprint, url_prefix="/admin")
app.secret_key = "attendance_secret_123"

UPLOAD_FOLDER = "data/class_photos"
OUTPUT_FOLDER = "output"
RESULTS_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

processor = AttendanceProcessor(threshold=0.4)
engine = FaceEngine(det_size=(1280, 1280), min_confidence=0.5)
matcher = Matcher(threshold=0.4)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def draw_on_photo(photo_path: str, output_path: str):
    img = cv2.imread(photo_path)
    faces = engine.get_embeddings(photo_path)
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        match = matcher.match(face["embedding"])
        color = (0, 200, 0) if match["matched"] else (0, 0, 220)
        label = match["name"].split("_")[-1].upper() if match["matched"] else "Unknown"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if match["matched"]:
            cv2.putText(img, f"{match['score']:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imwrite(output_path, img)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    import base64
    import re

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"class_{timestamp}.jpg"
    photo_path = os.path.join(UPLOAD_FOLDER, filename)

    # ── Check which method was used ───────────────────────
    photo_data = request.form.get("photo_data")  # camera capture
    file       = request.files.get("photo")       # file upload

    if photo_data:
        # ── Camera path ───────────────────────────────────
        # photo_data looks like:
        # "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
        # We strip the header and decode the base64 part

        try:
            # Remove "data:image/jpeg;base64," header
            header, encoded = photo_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)

            with open(photo_path, "wb") as f:
                f.write(img_bytes)

            logger.info(f"Camera photo saved: {photo_path}")

        except Exception as e:
            logger.error(f"Camera photo decode failed: {e}")
            flash("Could not process camera photo. Try again.")
            return redirect(url_for("index"))

    elif file and file.filename != "":
        # ── Upload path ───────────────────────────────────
        if not allowed_file(file.filename):
            flash("Only JPG and PNG files allowed")
            return redirect(url_for("index"))

        file.save(photo_path)
        logger.info(f"Uploaded photo saved: {photo_path}")

    else:
        flash("No photo provided")
        return redirect(url_for("index"))

    # ── Run attendance pipeline ───────────────────────────
    result = processor.process(photo_path)

    if not result:
        flash("Could not process photo. Make sure students are registered.")
        return redirect(url_for("index"))

    # ── Save CSV ──────────────────────────────────────────
    import csv as csv_module
    csv_filename = f"attendance_{timestamp}.csv"
    csv_path     = os.path.join(OUTPUT_FOLDER, csv_filename)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["date", "time", "student", "status", "score"]
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["attendance"]:
            writer.writerow({
                "date"   : row["date"],
                "time"   : row["time"],
                "student": row["student"],
                "status" : row["status"],
                "score"  : row["score"]
            })

    # ── Save annotated photo ──────────────────────────────
    annotated_filename = f"annotated_{timestamp}.jpg"
    annotated_path     = os.path.join(RESULTS_FOLDER, annotated_filename)
    draw_on_photo(photo_path, annotated_path)

    # ── Show results ──────────────────────────────────────
    return render_template(
        "results.html",
        result             = result,
        csv_filename       = csv_filename,
        annotated_filename = annotated_filename,
        enumerate          = enumerate
    )


@app.route("/history")
def history():
    files = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".csv")], reverse=True)
    return render_template("history.html", files=files)


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(os.path.abspath(OUTPUT_FOLDER), filename, as_attachment=True)


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Attendance System Starting...")
    print("  Open browser: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)