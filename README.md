# Face Recognition Attendance System

Automatically marks student attendance from a single class photo.

## Requirements
- Python 3.8 or higher
- Webcam or smartphone camera

## Setup

1. Clone project:
git clone https://github.com/YOUR_USERNAME/attendance_system.git
cd attendance_system

2. Create virtual environment:      
python -m venv venv

3. Activate:
venv\Scripts\activate

4. Install packages:
pip install -r requirements.txt

5. Add student photos to data/student_photos/

6. Register students:
python scripts/register_students.py

7. Run:
python app.py

8. Open browser: `http://localhost:5000`

## Admin Panel
URL: `http://localhost:5000/admin/login`  
Username: `admin`  
Password: `admin123`

## Built With
- InsightFace (RetinaFace + ArcFace)
- Flask
- OpenCV
- Python