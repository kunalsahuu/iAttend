from flask import Flask, render_template, request, redirect, send_file
import os
import cv2
import face_recognition
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from pathlib import Path

app = Flask(__name__)

# Constants
FACES_DIR = Path('known_faces')
ENCODINGS_FILE = Path('encodings.pkl')
CSV_FILE = Path('attendance.csv')
UPLOAD_FOLDER = Path('static/uploads')
# VIDEO_SOURCE = 0  # Webcam or IP camera stream
VIDEO_SOURCE = 'http://192.168.1.21:8080/video?320x240'
MIN_FACE_DISTANCE = 0.5  # Threshold for face recognition
FRAME_SCALE_FACTOR = 0.25  # Scale down frames for faster processing

# Create directories if they don't exist
FACES_DIR.mkdir(exist_ok=True)
UPLOAD_FOLDER.mkdir(exist_ok=True)

def load_or_encode_faces():
    """Load face encodings or generate them from known faces."""
    if ENCODINGS_FILE.exists():
        with ENCODINGS_FILE.open('rb') as f:
            data = pickle.load(f)
        return data['encodings'], data['names'], data.get('roll_nos', [])

    encodings, names, roll_nos = [], [], []
    
    for person_folder in FACES_DIR.iterdir():
        if not person_folder.is_dir():
            continue

        try:
            name, roll = person_folder.name.rsplit('_', 1)
        except ValueError:
            continue  # Skip folders not following name_roll format

        for img_file in person_folder.glob('*'):
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            face_locations = face_recognition.face_locations(img)
            if face_locations:
                encodings_list = face_recognition.face_encodings(img, face_locations)
                if encodings_list:
                    encodings.append(encodings_list[0])
                    names.append(name)
                    roll_nos.append(roll)

    with ENCODINGS_FILE.open('wb') as f:
        pickle.dump({'encodings': encodings, 'names': names, 'roll_nos': roll_nos}, f)

    return encodings, names, roll_nos

def mark_attendance(name, roll_no):
    """Record attendance with timestamp."""
    now = datetime.now()
    new_record = pd.DataFrame({
        'Name': [name],
        'Roll No': [roll_no],
        'Time': [now.strftime('%H:%M:%S')],
        'Date': [now.strftime('%Y-%m-%d')]
    })
    
    # Write to CSV with proper header handling
    header = not CSV_FILE.exists()
    new_record.to_csv(CSV_FILE, mode='a', header=header, index=False)

@app.route('/')
def index():
    """Display attendance records."""
    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        # Ensure consistent date formatting
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        data = df.values.tolist()
    else:
        data = []
    return render_template('index.html', data=data)

@app.route('/start')
def start_recognition():
    """Start face recognition and attendance marking."""
    known_encodings, known_names, known_rolls = load_or_encode_faces()
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    attendance = set()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame at lower resolution for faster recognition
            small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE_FACTOR, fy=FRAME_SCALE_FACTOR)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                # Calculate distances to known faces
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                # Find the best match
                min_distance = np.min(distances) if len(distances) > 0 else None
                
                if min_distance is not None and min_distance < MIN_FACE_DISTANCE:
                    best_match_index = np.argmin(distances)
                    roll_no = known_rolls[best_match_index]
                    
                    if roll_no not in attendance:
                        name = known_names[best_match_index]
                        mark_attendance(name, roll_no)
                        attendance.add(roll_no)

            cv2.imshow("Face Recognition - Press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return redirect('/')

@app.route('/upload', methods=['POST'])
def upload():
    """Upload new face photos for recognition."""
    name = request.form['name'].strip()
    roll_no = request.form['roll'].strip()
    files = request.files.getlist('photos')

    if not name or not roll_no:
        return "Name and Roll No are required", 400

    folder_path = FACES_DIR / f"{name}_{roll_no}"
    folder_path.mkdir(exist_ok=True)

    for file in files:
        if file.filename:  # Only save if file has a name
            file.save(str(folder_path / file.filename))

    # Invalidate cached encodings
    if ENCODINGS_FILE.exists():
        ENCODINGS_FILE.unlink()

    return redirect('/')

@app.route('/download')
def download():
    """Download attendance records as CSV."""
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