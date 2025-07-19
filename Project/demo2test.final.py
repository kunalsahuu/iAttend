from flask import Flask, render_template, request, redirect, send_file, session, url_for
import os
import cv2
import face_recognition
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from pathlib import Path


app = Flask(__name__)
app.secret_key = 'secret_key_here'

FACES_DIR = Path('known_faces')
ENCODINGS_FILE = Path('encodings.pkl')
CSV_FILE = Path('attendance.csv')
UPLOAD_FOLDER = Path('static/uploads')
VIDEO_SOURCE = 0
MIN_FACE_DISTANCE = 0.5
FRAME_SCALE_FACTOR = 0.25

FACES_DIR.mkdir(exist_ok=True)
UPLOAD_FOLDER.mkdir(exist_ok=True)

TEACHER_CREDENTIALS = {
    'samiksha': {'password': 'admin', 'name': 'Samiksha Shukla'},
    'kunal': {'password': 'admin1', 'name': 'Kunal kumar'},
    'himanshu': {'password': 'admin2', 'name': 'Himanshu Mokashe'},
    'priyanka': {'password': 'admin3', 'name': 'Priyanka Sahu'}
}

def load_or_encode_faces():
    if ENCODINGS_FILE.exists():
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
        return data['encodings'], data['names'], data.get('roll_nos', [])

    encodings, names, roll_nos = [], [], []

    for person_folder in FACES_DIR.iterdir():
        if person_folder.is_dir():
            try:
                name, roll = person_folder.name.rsplit('_', 1)
            except ValueError:
                continue

            for img_file in person_folder.glob('*'):
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                face_locations = face_recognition.face_locations(img)
                if face_locations:
                    enc = face_recognition.face_encodings(img, face_locations)
                    if enc:
                        encodings.append(enc[0])
                        names.append(name)
                        roll_nos.append(roll)

    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump({'encodings': encodings, 'names': names, 'roll_nos': roll_nos}, f)

    return encodings, names, roll_nos

def mark_attendance(name, roll_no):
    now = datetime.now()
    subject = session.get('subject', 'Unknown')
    teacher_name = session.get('teacher_name', 'Unknown')
    new_record = pd.DataFrame({
        'Name': [name],
        'Roll No': [roll_no],
        'Time': [now.strftime('%H:%M:%S')],
        'Date': [now.strftime('%Y-%m-%d')],
        'Subject': [subject],
        'Teacher': [teacher_name]
    })
    new_record.to_csv(CSV_FILE, mode='a', header=not CSV_FILE.exists(), index=False)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['POST'])
def login():
    role = request.form.get('role')
    if role == 'teacher':
        return redirect(url_for('teacher_login'))
    elif role == 'student':
        return redirect(url_for('student_view'))
    return redirect('/')

@app.route('/teacher_login', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        teacher = TEACHER_CREDENTIALS.get(username)
        if teacher and teacher['password'] == password:
            session['teacher'] = True
            session['teacher_name'] = teacher['name']
            return redirect(url_for('select_subject'))
        else:
            return 'Invalid credentials', 401
    return render_template('teacher_login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

@app.route('/select_subject', methods=['GET', 'POST'])
def select_subject():
    if request.method == 'POST':
        subject = request.form.get('subject')
        if subject:
            session['subject'] = subject
            return redirect(url_for('start_recognition'))
    return render_template('select_subject.html')

@app.route('/student')
def student_view():
    data = []
    summary = []
    subject_max_counts = {}

    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        data = df.values.tolist()

        # Get attendance count per student per subject
        summary_df = df.groupby(['Name', 'Roll No', 'Subject']).size().unstack(fill_value=0)

        # Get total classes conducted per subject
        subject_max_counts = summary_df.max().to_dict()

        # Calculate percentage columns
        for subject in subject_max_counts:
            max_count = subject_max_counts[subject]
            if max_count > 0:
                summary_df[subject + ' %'] = (summary_df[subject] / max_count * 100).round(1)

        summary_df['Total'] = summary_df[[subj for subj in subject_max_counts]].sum(axis=1)
        summary_df = summary_df.reset_index()
        summary = summary_df.to_dict(orient='records')

    return render_template(
        'student.html',
        records=data,
        summary=summary,
        subject_max_counts=subject_max_counts
    )



@app.route('/index')
def index():
    if not session.get('teacher'):
        return redirect(url_for('teacher_login'))

    data = []
    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        data = df.values.tolist()
    return render_template('index.html', data=data, teacher=session.get('teacher_name'))

@app.route('/start')
def start_recognition():
    known_encodings, known_names, known_rolls = load_or_encode_faces()
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    attendance = set()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE_FACTOR, fy=FRAME_SCALE_FACTOR)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                min_distance = np.min(distances) if len(distances) > 0 else None

                # Scale back up face location
                top, right, bottom, left = [int(pos / FRAME_SCALE_FACTOR) for pos in face_location]

                if min_distance is not None and min_distance < MIN_FACE_DISTANCE:
                    best_match_index = np.argmin(distances)
                    name = known_names[best_match_index]
                    roll_no = known_rolls[best_match_index]

                    if roll_no not in attendance:
                        mark_attendance(name, roll_no)
                        attendance.add(roll_no)

                    label = f"Name: {name} | Roll No: {roll_no}"
                    color = (0, 255, 0)  # Green for recognized
                else:
                    label = "Unknown"
                    color = (0, 0, 255)  # Red for unrecognized

                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Background rectangle for text
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = left
                text_y = bottom + 20
                cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 4), 
                              (text_x + text_size[0] + 2, text_y + 4), color, cv2.FILLED)

                # Put text label
                cv2.putText(frame, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show feed
            cv2.imshow("Face Recognition - Press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return redirect(url_for('index'))


@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['name'].strip()
    roll_no = request.form['roll'].strip()
    files = request.files.getlist('photos')

    if not name or not roll_no:
        return "Name and Roll No are required", 400

    folder_path = FACES_DIR / f"{name}_{roll_no}"
    folder_path.mkdir(exist_ok=True)

    for file in files:
        if file.filename:
            file.save(str(folder_path / file.filename))

    if ENCODINGS_FILE.exists():
        ENCODINGS_FILE.unlink()

    return redirect(url_for('index'))

@app.route('/download')
def download():
    if not CSV_FILE.exists():
        return "No attendance records found", 404

    return send_file(
        str(CSV_FILE),
        as_attachment=True,
        download_name='attendance.csv',
        mimetype='text/csv'
    )

if __name__ == '__main__':
    app.run(debug=True)
