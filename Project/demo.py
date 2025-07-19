from flask import Flask, render_template, request, redirect, send_file, session, url_for, flash
import os
import cv2
import face_recognition
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your_very_strong_secret_key_here'  # Change this to a random string

# Configuration
FACES_DIR = Path('known_faces')
ENCODINGS_FILE = Path('encodings.pkl')
CSV_FILE = Path('attendance.csv')
UPLOAD_FOLDER = Path('static/uploads')
SUMMARY_CSV_FILE = Path("static/summary.csv")
VIDEO_SOURCE = 0  # Default webcam (can be IP camera URL)
MIN_FACE_DISTANCE = 0.6  # Increased tolerance for far faces
FRAME_SCALE_FACTOR = 0.5  # Increased from 0.25 for better far detection
MIN_FACE_SIZE = 100  # Minimum face size in pixels to process
DETECTION_MODEL = "hog"  # "hog" (faster) or "cnn" (more accurate)

# Create directories
FACES_DIR.mkdir(exist_ok=True)
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Teacher credentials
TEACHER_CREDENTIALS = {
    'samiksha': {'password': 'admin', 'name': 'Samiksha Shukla'},
    'kunal': {'password': 'admin1', 'name': 'Kunal kumar'},
    'himanshu': {'password': 'admin2', 'name': 'Himanshu Mokashe'},
    'priyanka': {'password': 'admin3', 'name': 'Priyanka Sahu'}
}

def load_or_encode_faces():
    """Load or generate face encodings with enhanced detection"""
    if ENCODINGS_FILE.exists():
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
            return data['encodings'], data['names'], data.get('roll_nos', [])
        except Exception as e:
            print(f"Error loading encodings: {e}")
            os.remove(ENCODINGS_FILE)

    encodings, names, roll_nos = [], [], []

    for person_folder in FACES_DIR.iterdir():
        if person_folder.is_dir():
            try:
                name, roll = person_folder.name.rsplit('_', 1)
            except ValueError:
                continue

            for img_file in person_folder.glob('*'):
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Use specified model for detection
                    face_locations = face_recognition.face_locations(
                        rgb_img, 
                        model=DETECTION_MODEL,
                        number_of_times_to_upsample=1
                    )
                    
                    if face_locations:
                        enc = face_recognition.face_encodings(rgb_img, face_locations)
                        if enc:
                            encodings.append(enc[0])
                            names.append(name)
                            roll_nos.append(roll)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    continue

    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump({'encodings': encodings, 'names': names, 'roll_nos': roll_nos}, f)

    return encodings, names, roll_nos

def mark_attendance(name, roll_no):
    """Record attendance with timestamp and validation"""
    now = datetime.now()
    subject = session.get('subject', 'Unknown')
    teacher_name = session.get('teacher_name', 'Unknown')
    
    # Check for existing record today
    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        today = now.strftime('%Y-%m-%d')
        existing = df[(df['Name'] == name) & 
                     (df['Roll No'] == roll_no) & 
                     (df['Date'] == today) & 
                     (df['Subject'] == subject)]
        if not existing.empty:
            return False
    
    # Create new record
    new_record = pd.DataFrame({
        'Name': [name],
        'Roll No': [roll_no],
        'Time': [now.strftime('%H:%M:%S')],
        'Date': [now.strftime('%Y-%m-%d')],
        'Subject': [subject],
        'Teacher': [teacher_name]
    })
    
    new_record.to_csv(CSV_FILE, mode='a', header=not CSV_FILE.exists(), index=False)
    return True

@app.route('/')
def landing():
    """Landing page with role selection"""
    return render_template('landing.html')

@app.route('/login', methods=['POST'])
def login():
    """Handle login role selection"""
    role = request.form.get('role')
    if role == 'teacher':
        return redirect(url_for('teacher_login'))
    elif role == 'student':
        return redirect(url_for('student_view'))
    flash('Please select a valid role', 'warning')
    return redirect(url_for('landing'))

@app.route('/teacher_login', methods=['GET', 'POST'])
def teacher_login():
    """Teacher login page"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            flash('Username and password are required', 'danger')
            return redirect(url_for('teacher_login'))
        
        teacher = TEACHER_CREDENTIALS.get(username)
        if teacher and teacher['password'] == password:
            session['teacher'] = True
            session['teacher_name'] = teacher['name']
            flash('Login successful', 'success')
            return redirect(url_for('select_subject'))
        
        flash('Invalid credentials', 'danger')
    
    return render_template('teacher_login.html')

@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('landing'))

@app.route('/select_subject', methods=['GET', 'POST'])
def select_subject():
    """Subject selection page"""
    if not session.get('teacher'):
        flash('Please login first', 'warning')
        return redirect(url_for('teacher_login'))
    
    if request.method == 'POST':
        subject = request.form.get('subject')
        if subject:
            session['subject'] = subject
            flash(f'Subject "{subject}" selected', 'success')
            return redirect(url_for('index'))
        flash('Please select a subject', 'danger')
    
    return render_template('select_subject.html')

@app.route('/student')
def student_view():
    """Student view with attendance summary"""
    data = []
    summary = []
    subject_max_counts = {}
    top_5_students = []

    try:
        if CSV_FILE.exists():
            df = pd.read_csv(CSV_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            data = df.values.tolist()

            # Generate summary statistics
            summary_df = df.groupby(['Name', 'Roll No', 'Subject']).size().unstack(fill_value=0)
            subject_max_counts = summary_df.max().to_dict()

            for subject in subject_max_counts:
                max_count = subject_max_counts[subject]
                if max_count > 0:
                    summary_df[subject + ' %'] = (summary_df[subject] / max_count * 100).round(1)

            summary_df['Total'] = summary_df[[subj for subj in subject_max_counts]].sum(axis=1)
            summary_df['Max Total'] = sum(subject_max_counts.values())
            summary_df['Total %'] = (summary_df['Total'] / summary_df['Max Total'] * 100).round(1)

            summary_df = summary_df.reset_index()
            summary_df.to_csv(SUMMARY_CSV_FILE, index=False)

            sorted_summary = summary_df.sort_values(by='Total', ascending=False)
            top_5_students = sorted_summary.head(5).to_dict(orient='records')
            summary = sorted_summary.to_dict(orient='records')
    except Exception as e:
        flash('Error loading attendance data', 'danger')
        print(f"Error in student_view: {e}")

    return render_template(
        'student.html',
        records=data,
        summary=summary,
        subject_max_counts=subject_max_counts,
        top_5_students=top_5_students
    )

@app.route('/index')
def index():
    """Main teacher dashboard"""
    if not session.get('teacher'):
        flash('Please login first', 'warning')
        return redirect(url_for('teacher_login'))
    
    if 'subject' not in session:
        flash('Please select a subject first', 'warning')
        return redirect(url_for('select_subject'))
    
    data = []
    try:
        if CSV_FILE.exists():
            df = pd.read_csv(CSV_FILE)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            current_subject = session.get('subject')
            data = df[df['Subject'] == current_subject].values.tolist()
    except Exception as e:
        flash('Error loading attendance data', 'danger')
        print(f"Error in index: {e}")
    
    return render_template(
        'index.html',
        data=data,
        teacher=session.get('teacher_name'),
        current_subject=session.get('subject')
    )

@app.route('/start')
def start_recognition():
    """Enhanced face recognition with far-distance capabilities"""
    if not session.get('teacher'):
        flash('Please login first', 'warning')
        return redirect(url_for('teacher_login'))
    
    if 'subject' not in session:
        flash('Please select a subject first', 'warning')
        return redirect(url_for('select_subject'))
    
    known_encodings, known_names, known_rolls = load_or_encode_faces()
    if not known_encodings:
        flash('No registered faces found', 'danger')
        return redirect(url_for('index'))
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        flash('Could not open video source', 'danger')
        return redirect(url_for('index'))
    
    # Configure camera for better far detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_FOCUS, 0)  # 0 = infinity focus
    
    attendance = set()
    process_this_frame = True
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE_FACTOR, fy=FRAME_SCALE_FACTOR)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(
                    rgb_small_frame,
                    model=DETECTION_MODEL,
                    number_of_times_to_upsample=2
                )

                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    top = int(top / FRAME_SCALE_FACTOR)
                    right = int(right / FRAME_SCALE_FACTOR)
                    bottom = int(bottom / FRAME_SCALE_FACTOR)
                    left = int(left / FRAME_SCALE_FACTOR)

                    face_size = (right - left) * (bottom - top)
                    if face_size < MIN_FACE_SIZE:
                        continue

                    matches = face_recognition.compare_faces(
                        known_encodings, 
                        face_encoding, 
                        tolerance=MIN_FACE_DISTANCE
                    )
                    name = "Unknown"
                    roll_no = ""

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_names[first_match_index]
                        roll_no = known_rolls[first_match_index]

                        if roll_no not in attendance:
                            if mark_attendance(name, roll_no):
                                attendance.add(roll_no)
                                flash(f'Attendance marked for {name} ({roll_no})', 'success')

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    label = f"{name} ({roll_no})" if roll_no else name
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(frame, label, (left + 6, bottom - 6), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            process_this_frame = not process_this_frame

            cv2.imshow(f"Face Recognition - {session.get('subject', '')} (Press Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        flash(f'Recognition error: {str(e)}', 'danger')
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    """Handle student registration with photos"""
    if not session.get('teacher'):
        flash('Please login first', 'warning')
        return redirect(url_for('teacher_login'))
    
    name = request.form.get('name', '').strip()
    roll_no = request.form.get('roll', '').strip()
    files = request.files.getlist('photos')

    if not name or not roll_no:
        flash('Name and Roll No are required', 'danger')
        return redirect(url_for('index'))
    
    if not files or all(file.filename == '' for file in files):
        flash('Please upload at least one photo', 'danger')
        return redirect(url_for('index'))

    try:
        folder_path = FACES_DIR / f"{name}_{roll_no}"
        folder_path.mkdir(exist_ok=True)

        for file in files:
            if file.filename:
                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                file.save(str(folder_path / filename))

        if ENCODINGS_FILE.exists():
            os.remove(ENCODINGS_FILE)
        
        flash(f'Student {name} registered successfully', 'success')
    except Exception as e:
        flash(f'Registration failed: {str(e)}', 'danger')
    
    return redirect(url_for('index'))

@app.route('/download')
def download():
    """Download full attendance CSV"""
    if not session.get('teacher'):
        flash('Please login first', 'warning')
        return redirect(url_for('teacher_login'))
    
    if not CSV_FILE.exists():
        flash('No attendance records found', 'danger')
        return redirect(url_for('index'))
    
    subject = session.get('subject', 'all')
    return send_file(
        str(CSV_FILE),
        as_attachment=True,
        download_name=f'attendance_{subject}.csv',
        mimetype='text/csv'
    )

@app.route('/download_summary')
def download_summary():
    """Download summary statistics CSV"""
    if not SUMMARY_CSV_FILE.exists():
        flash('Summary data not available', 'danger')
        return redirect(url_for('student_view'))
    
    return send_file(
        str(SUMMARY_CSV_FILE),
        as_attachment=True,
        download_name='attendance_summary.csv',
        mimetype='text/csv'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)