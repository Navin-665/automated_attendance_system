# Admin backend routes for student and attendance management


import os
import sys
import csv
import pickle
import shutil
import logging
import numpy as np
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, send_file

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.face_engine import FaceEngine
from modules.matcher import Matcher

logger = logging.getLogger(__name__)

admin = Blueprint("admin", __name__)

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"
EMBEDDINGS_FILE = "data/embeddings.pkl"
STUDENT_DIR = "data/student_photos"
OUTPUT_DIR = "output"
THUMBS_DIR = "static/student_thumbs"

os.makedirs(THUMBS_DIR, exist_ok=True)

engine = FaceEngine(det_size=(640, 640), min_confidence=0.6)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def load_embeddings() -> dict:
    if not os.path.exists(EMBEDDINGS_FILE):
        return {}
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)


def save_embeddings(embeddings: dict):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)


def load_all_attendance() -> list:
    records = []
    record_id = 0

    if not os.path.exists(OUTPUT_DIR):
        return records

    for filename in sorted(os.listdir(OUTPUT_DIR), reverse=True):
        if not filename.endswith(".csv"):
            continue

        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "id": record_id,
                    "date": row.get("date", ""),
                    "time": row.get("time", ""),
                    "student": row.get("student", ""),
                    "status": row.get("status", ""),
                    "score": safe_float(row.get("score", 0)),
                    "file": filename
                })
                record_id += 1

    return records


def get_students() -> list:
    students = []
    embeddings = load_embeddings()

    if not os.path.exists(STUDENT_DIR):
        return students

    for folder_name in sorted(os.listdir(STUDENT_DIR)):
        folder_path = os.path.join(STUDENT_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        photos = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        thumb = None
        if photos:
            src = os.path.join(folder_path, photos[0])
            dest = os.path.join(THUMBS_DIR, f"{folder_name}.jpg")
            if not os.path.exists(dest):
                shutil.copy(src, dest)
            thumb = f"{folder_name}.jpg"

        parts = folder_name.split("_")
        name = parts[-1].capitalize() if parts else folder_name
        roll = parts[1] if len(parts) >= 2 else "—"

        students.append({
            "folder": folder_name,
            "name": name,
            "roll": roll,
            "photo_count": len(photos),
            "photo": thumb,
            "registered": folder_name in embeddings
        })

    return students


def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin.login"))
        return f(*args, **kwargs)
    return decorated


@admin.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin.dashboard"))
        error = "Wrong username or password"
    return render_template("admin/login.html", error=error)


@admin.route("/logout")
def logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin.login"))


# ── Dashboard ─────────────────────────────────────────────

@admin.route("/dashboard")
@login_required
def dashboard():
    records = load_all_attendance()
    students = get_students()
    today = datetime.now().strftime("%Y-%m-%d")
    
    grouped = {}
    for record in records:
        key = (record["date"], record["time"], record["file"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(record)
    
    sessions = []
    for (date, time, filename), session_records in grouped.items():
        sessions.append({
            "date": date,
            "time": time,
            "filename": filename,
            "records": session_records,
            "present_count": sum(1 for r in session_records if r["status"] == "PRESENT"),
            "total_count": len(session_records)
        })
    
    sessions.sort(key=lambda x: (x["date"], x["time"]), reverse=True)
    
    today_sessions = [s for s in sessions if s["date"] == today]
    today_present = sum(s["present_count"] for s in today_sessions)
    
    return render_template("admin/dashboard.html", total_students=len(students), total_records=len(records),
                          today_present=today_present, total_sessions=len(sessions), recent_sessions=sessions[:3])


@admin.route("/students")
@login_required
def students():
    return render_template("admin/students.html", students=get_students())


@admin.route("/students/add", methods=["POST"])
@login_required
def add_student():
    name = request.form.get("name", "").strip().lower()
    roll = request.form.get("roll", "").strip()
    photos = request.files.getlist("photos")

    if not name or not roll:
        flash("Name and roll number required", "error")
        return redirect(url_for("admin.students"))

    if not photos or all(p.filename == "" for p in photos):
        flash("Please upload at least one photo", "error")
        return redirect(url_for("admin.students"))

    folder_name = f"student_{roll}_{name}"
    folder_path = os.path.join(STUDENT_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    saved = 0
    for i, photo in enumerate(photos):
        if photo.filename == "":
            continue
        ext = photo.filename.rsplit(".", 1)[-1].lower()
        photo.save(os.path.join(folder_path, f"photo{i+1}.{ext}"))
        saved += 1

    if saved == 0:
        flash("No valid photos saved", "error")
        return redirect(url_for("admin.students"))

    embeddings = load_embeddings()
    student_embs = []

    for photo_file in os.listdir(folder_path):
        faces = engine.get_embeddings(os.path.join(folder_path, photo_file))
        if len(faces) == 1:
            student_embs.append(faces[0]["embedding"])

    if student_embs:
        avg = np.mean(student_embs, axis=0)
        avg = avg / np.linalg.norm(avg)
        embeddings[folder_name] = avg
        save_embeddings(embeddings)
        flash(f"✅ {name.capitalize()} registered with {saved} photos", "success")
    else:
        flash(f"⚠ Photos saved but no faces detected. Re-upload clearer photos.", "error")

    return redirect(url_for("admin.students"))


@admin.route("/students/delete/<folder_name>")
@login_required
def delete_student(folder_name):
    folder_path = os.path.join(STUDENT_DIR, folder_name)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    embeddings = load_embeddings()
    if folder_name in embeddings:
        del embeddings[folder_name]
        save_embeddings(embeddings)

    thumb = os.path.join(THUMBS_DIR, f"{folder_name}.jpg")
    if os.path.exists(thumb):
        os.remove(thumb)

    flash(f"Student {folder_name} deleted", "success")
    return redirect(url_for("admin.students"))


@admin.route("/attendance")
@login_required
def attendance():
    records = load_all_attendance()
    grouped = {}
    
    for record in records:
        key = (record["date"], record["time"], record["file"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(record)
    
    sessions = []
    for (date, time, filename), session_records in grouped.items():
        sessions.append({
            "date": date,
            "time": time,
            "filename": filename,
            "records": session_records,
            "present_count": sum(1 for r in session_records if r["status"] == "PRESENT"),
            "total_count": len(session_records)
        })
    
    sessions.sort(key=lambda x: (x["date"], x["time"]), reverse=True)
    return render_template("admin/attendance.html", sessions=sessions)


@admin.route("/edit/<int:record_id>", methods=["GET", "POST"])
@login_required
def edit_record(record_id):
    if request.method == "GET":
        return redirect(url_for("admin.attendance"))
    
    new_status = request.form.get("status")
    if new_status not in ("PRESENT", "ABSENT"):
        return redirect(url_for("admin.attendance"))

    records = load_all_attendance()
    if record_id >= len(records):
        flash("Record not found", "error")
        return redirect(url_for("admin.attendance"))

    record = records[record_id]
    filepath = os.path.join(OUTPUT_DIR, record["file"])

    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        if (row["student"] == record["student"] and
            row["date"] == record["date"] and
            row["time"] == record["time"]):
            row["status"] = new_status
            break

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    flash(f"Updated {record['student']} to {new_status}", "success")
    return redirect(url_for("admin.attendance"))


@admin.route("/export")
@login_required
def export_csv():
    records = load_all_attendance()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(OUTPUT_DIR, f"attendance_export_{timestamp}.csv")

    with open(export_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "time", "student", "status", "score"])
        writer.writeheader()
        for record in records:
            writer.writerow({
                "date": record["date"],
                "time": record["time"],
                "student": record["student"],
                "status": record["status"],
                "score": record["score"] if record["score"] > 0 else "—"
            })

    return send_file(os.path.abspath(export_path), as_attachment=True, download_name=f"attendance_{timestamp}.csv")