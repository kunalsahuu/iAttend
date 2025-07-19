from flask import Flask, render_template, request, redirect, send_file
import os
import cv2
import face_recognition
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

app = Flask(__name__)

FACES_DIR = 'known_faces'
ENCODINGS_FILE = 'encodings.pkl'
CSV_FILE = 'attendance.csv'
UPLOAD_FOLDER = 'static/uploads'
VIDEO_SOURCE = 0  # Webcam or IP camera stream

if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_or_encode_faces():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
        known_encodings = data['encodings']
        known_names = data['names']
        known_rolls = data.get('roll_nos', ['Unknown'] * len(known_names))
        return known_encodings, known_names, known_rolls

    encodings, names, roll_nos = [], [], []
    for person_folder in os.listdir(FACES_DIR):
        person_path = os.path.join(FACES_DIR, person_folder)
        if not os.path.isdir(person_path):
            continue

        try:
            name, roll = person_folder.rsplit('_', 1)
        except ValueError:
            continue  # Skip folders not following name_roll format

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
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
    time_str = now.strftime('%H:%M:%S')
    date_str = now.strftime('%Y-%m-%d')
    
    # Create a new DataFrame with proper date formatting
    df = pd.DataFrame({
        'Name': [name],
        'Roll No': [roll_no],
        'Time': [time_str],
        'Date': [date_str]  # Already formatted as string
    })
    
    # Write to CSV with proper handling
    if not os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, index=False)
    else:
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)

@app.route('/')
def index():
    data = []
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        # Convert date strings to datetime objects for consistent display
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        data = df.values.tolist()
    return render_template('index.html', data=data)

@app.route('/start')
def start_recognition():
    known_encodings, known_names, known_rolls = load_or_encode_faces()
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    attendance = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding in encodings:
            distances = face_recognition.face_distance(known_encodings, encoding)
            if len(distances) == 0:
                continue

            min_index = np.argmin(distances)
            if distances[min_index] < 0.5:
                name = known_names[min_index]
                roll = known_rolls[min_index]
                if roll not in attendance:
                    mark_attendance(name, roll)
                    attendance.add(roll)

        cv2.imshow("Press 'q' to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect('/')

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['name'].strip()
    roll = request.form['roll'].strip()
    files = request.files.getlist('photos')

    if not name or not roll:
        return "Name and Roll No required", 400

    folder_name = f"{name}_{roll}"
    folder_path = os.path.join(FACES_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    for file in files:
        file.save(os.path.join(folder_path, file.filename))

    if os.path.exists(ENCODINGS_FILE):
        os.remove(ENCODINGS_FILE)

    return redirect('/')

@app.route('/download')
def download():
    # Ensure the CSV is properly formatted before downloading
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        # Convert dates to proper format if needed
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        # Save to a temporary file
        temp_csv = 'temp_attendance.csv'
        df.to_csv(temp_csv, index=False)
        return send_file(temp_csv, as_attachment=True, download_name='attendance.csv')
    return "No attendance records found", 404

if __name__ == '__main__':
    app.run(debug=True)