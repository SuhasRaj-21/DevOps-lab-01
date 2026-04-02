import cv2
import sqlite3
import datetime
import os

def mark_attendance(student_id, name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    date_today = datetime.datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.datetime.now().strftime("%H:%M:%S")

    c.execute("SELECT * FROM attendance WHERE id=? AND date=?", (student_id, date_today))
    if not c.fetchone():
        c.execute("INSERT INTO attendance (id, name, date, time) VALUES (?, ?, ?, ?)", (student_id, name, date_today, time_now))
        conn.commit()
    conn.close()

def recognize_faces():
    if not os.path.exists('trainer.yml'):
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')

    cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX

    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT id, name FROM students")
    students = dict(c.fetchall())
    conn.close()

    cam = cv2.VideoCapture(0)
    
    while True:
        ret, im = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id_pred, distance = recognizer.predict(gray[y:y+h, x:x+w])

            if distance < 80:
                name = students.get(id_pred, "Unknown")
                if name != "Unknown":
                    mark_attendance(id_pred, name)
            else:
                name = "Unknown"
            
            cv2.putText(im, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

        cv2.imshow('Face Recognition (Press ESC to stop)', im)

        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cam.release()
    cv2.destroyAllWindows()
