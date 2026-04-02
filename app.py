from flask import Flask, render_template, request, redirect, flash, url_for
import sqlite3
import os
import face_capture
import train
import recognize

app = Flask(__name__)
app.secret_key = "attendance_secret_key"

def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, name TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (id INTEGER, name TEXT, date TEXT, time TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC")
    records = c.fetchall()
    conn.close()
    return render_template('index.html', records=records)

@app.route('/capture', methods=['POST'])
def capture():
    student_id = request.form.get('student_id')
    student_name = request.form.get('student_name')
    if not student_id or not student_name:
        flash("Student ID and Name required!")
        return redirect(url_for('index'))
    
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM students WHERE id=?", (student_id,))
    if not c.fetchone():
        c.execute("INSERT INTO students (id, name) VALUES (?, ?)", (student_id, student_name))
        conn.commit()
    conn.close()
    
    face_capture.capture_faces(student_id)
    flash("Faces captured successfully!")
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train_model():
    train.train_classifier()
    flash("Model trained successfully!")
    return redirect(url_for('index'))

@app.route('/start_attendance', methods=['POST'])
def start_attendance():
    recognize.recognize_faces()
    flash("Attendance successfully recorded!")
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    app.run(debug=True, port=5000)
