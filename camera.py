import cv2
import mediapipe as mp
import numpy as np
import math
import serial
import serial.tools.list_ports
import time

# Function to find Arduino port
def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if 'Arduino' in port.description or 'CH340' in port.description or 'USB Serial' in port.description:
            return port.device
    return None

# Attempt to connect to Arduino
arduino = None
try:
    arduino_port = find_arduino_port()
    if arduino_port:
        print(f"Found Arduino on port: {arduino_port}")
        arduino = serial.Serial(arduino_port, 9600, timeout=1)
        time.sleep(2)
        print("Connected to Arduino")
    else:
        print("Arduino not found! Check the connection.")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    arduino = None

# Mediapipe setups
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
Output = 0
# Mediapipe face mesh configuration
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Mediapipe hands configuration
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Helper functions
def send_to_arduino(command):
    if arduino is not None:
        try:
            arduino.write(bytes([command]))
        except:
            print("Error sending to Arduino!")

def calculate_distance(p1, p2, img):
    x1, y1 = int(p1.x * img.shape[1]), int(p1.y * img.shape[0])
    x2, y2 = int(p2.x * img.shape[1]), int(p2.y * img.shape[0])
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def is_finger_up(finger_points, hand_landmarks):
    base = hand_landmarks.landmark[finger_points[0]]
    tip = hand_landmarks.landmark[finger_points[3]]
    return tip.y < base.y

def detect_gesture(hand_landmarks):
    global Output, PreviousCommand
    index_up = is_finger_up([5, 6, 7, 8], hand_landmarks)
    middle_up = is_finger_up([9, 10, 11, 12], hand_landmarks)
    ring_up = is_finger_up([13, 14, 15, 16], hand_landmarks)
    pinky_up = is_finger_up([17, 18, 19, 20], hand_landmarks)

    thumb_tip = hand_landmarks.landmark[4]
    wrist = hand_landmarks.landmark[0]

    distances = [
        math.sqrt((thumb_tip.x - wrist.x) ** 2 + (thumb_tip.y - wrist.y) ** 2)
    ]
    avg_distance = sum(distances) / len(distances)

    PreviousCommand = Output
    if avg_distance < 0.2:
        Output = 1
        gesture = "Fist"
    elif index_up and middle_up and not ring_up and not pinky_up:
        Output = 2
        gesture = "Peace"
    else:
        Output = 0
        gesture = "Other"

    if Output != PreviousCommand:
        send_to_arduino(Output)

    return gesture

# Video capture setup
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face mesh detection
    face_results = face_mesh.process(img_rgb)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    # Hand detection
    hand_results = hands.process(img_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            gesture = detect_gesture(hand_landmarks)

            thumb_index_distance = calculate_distance(
                hand_landmarks.landmark[4],
                hand_landmarks.landmark[8],
                img
            )

            cv2.putText(img, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Thumb-Index Distance: {int(thumb_index_distance)}px", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Combined Detector", img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

if arduino is not None:
    arduino.close()
cap.release()
cv2.destroyAllWindows()
