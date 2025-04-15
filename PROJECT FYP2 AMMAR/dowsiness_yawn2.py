from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import sqlite3
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD

# ---- GPIO SETUP ----
BUZZER_PIN = 20  
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

BLUE_LED = 23     # For eye drowsiness detection
WHITE_LED = 24    # For yawn detection
GPIO.setup(BLUE_LED, GPIO.OUT)
GPIO.setup(WHITE_LED, GPIO.OUT)

# ---- LCD SETUP ----
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, charmap='A00')

# database setup
def log_to_database(status, buzzer_status, camera_status):
    # Get the system uptime (you can adjust this depending on how you want to track uptime)
    system_uptime = time.time()

    # Connect to the SQLite database
    conn = sqlite3.connect('sleep_log.db')
    cur = conn.cursor()

    # Insert the data into the database
    cur.execute('''
    INSERT INTO drowsiness_logs (status, system_uptime, buzzer_status, camera_status)
    VALUES (?, ?, ?, ?)
    ''', (status, system_uptime, buzzer_status, camera_status))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# ---- Eye & Yawn Detection Functions ----
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    return ((eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0, leftEye, rightEye)

def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    return abs(np.mean(top_lip, axis=0)[1] - np.mean(low_lip, axis=0)[1])

# ---- Parameters ----
EYE_AR_THRESH = 0.29
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
COUNTER = 0

# ---- Load Face Detector & Landmark Predictor ----
print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# ---- Start Video Stream ----
print("-> Starting Video Stream")
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, leftEye, rightEye = final_ear(shape)
        distance = lip_distance(shape)

        # Draw Eyes & Lips
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [shape[48:60]], -1, (0, 255, 0), 1)

        # ---- Drowsiness Detection ----
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Activate Buzzer
                GPIO.output(BLUE_LED, GPIO.HIGH)        # Blue LED ON
                lcd.clear()
                lcd.write_string("Drowsiness!")
                log_to_database('Drowsiness Detected', 'ON', 'Active') # Log to the database
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            GPIO.output(BUZZER_PIN, GPIO.LOW)  # Turn off Buzzer
            GPIO.output(BLUE_LED, GPIO.LOW)    # Blue LED OFF
            lcd.clear()

        # ---- Yawn Detection ----
        if distance > YAWN_THRESH:
            GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Activate Buzzer
            GPIO.output(WHITE_LED, GPIO.HIGH)           # White LED ON
            lcd.clear()
            lcd.write_string("Yawning Detected!")
            log_to_database('Yawning Detected', 'ON', 'Active') # Log to the database
            cv2.putText(frame, "YAWN ALERT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            GPIO.output(BUZZER_PIN, GPIO.LOW)  # Turn off Buzzer
            GPIO.output(WHITE_LED, GPIO.LOW)   # White LED OFF

        # Display EAR & Yawn Distance
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show Video Stream
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
lcd.clear()
GPIO.output(BUZZER_PIN, GPIO.LOW)
GPIO.cleanup()
