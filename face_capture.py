import cv2
import os

def capture_faces(student_id):
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
        
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(f"dataset/User.{student_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.imshow('Face Capture (Press ESC or wait for 50 captures)', img)

        if cv2.waitKey(100) & 0xFF == 27:
            break
        elif count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
