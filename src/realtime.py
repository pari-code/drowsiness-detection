import cv2
import torch
import numpy as np
import mediapipe as mp
import pygame
from collections import deque
from pathlib import Path
from PIL import Image
from torchvision import transforms
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DrowsinessDetector

MODEL_PATH      = "weights/best_model.pth"
ALARM_PATH      = "alarm.mp3"
SEQ_LEN         = 4        # must match training seq_len
EAR_THRESHOLD   = 0.25
MAR_THRESHOLD   = 0.65
MODEL_THRESHOLD = 0.70
CONSEC_FRAMES   = 20

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61,  291, 0,   17,  13,  14 ]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )
])


def eye_aspect_ratio(landmarks, eye_pts, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_pts])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def mouth_aspect_ratio(landmarks, mouth_pts, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in mouth_pts])
    vertical   = np.linalg.norm(pts[2] - pts[3]) + np.linalg.norm(pts[4] - pts[5])
    horizontal = np.linalg.norm(pts[0] - pts[1]) + 1e-6
    return vertical / (2.0 * horizontal)


def crop_face(frame, landmarks, w, h, padding=0.12):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1 = max(0, int(min(xs) - padding * w))
    y1 = max(0, int(min(ys) - padding * h))
    x2 = min(w, int(max(xs) + padding * w))
    y2 = min(h, int(max(ys) + padding * h))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else frame


class DrowsinessMonitor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on: {self.device}")

        ckpt = torch.load(MODEL_PATH, map_location=self.device)
        self.model = DrowsinessDetector(seq_len=SEQ_LEN).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"Model loaded — val_f1={ckpt['val_f1']:.4f}")

        self.mp_mesh  = mp.solutions.face_mesh
        self.face_mesh = self.mp_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.frame_buffer = deque(maxlen=SEQ_LEN)
        self.alert_counter = 0
        self.alarm_on      = False
        self.eye_states    = deque(maxlen=300)

        pygame.mixer.init()
        if Path(ALARM_PATH).exists():
            pygame.mixer.music.load(ALARM_PATH)
        else:
            print(f"Warning: {ALARM_PATH} not found — alarm disabled")

    def predict(self):
        if len(self.frame_buffer) < SEQ_LEN:
            return 0.0
        seq = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.softmax(self.model(seq), dim=1)[0, 1].item()
        return prob

    def trigger_alarm(self):
        if not self.alarm_on:
            if Path(ALARM_PATH).exists():
                pygame.mixer.music.play(-1)
            self.alarm_on = True

    def stop_alarm(self):
        if self.alarm_on:
            pygame.mixer.music.stop()
            self.alarm_on = False

    def draw_hud(self, frame, ear, mar, drowsy_prob, perclos):
        h, w = frame.shape[:2]
        color = (0, 50, 255) if self.alarm_on else (50, 200, 80)
        cv2.putText(frame, f"EAR: {ear:.2f}",           (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"MAR: {mar:.2f}",           (10, 58),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"Drowsy: {drowsy_prob*100:.0f}%", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"PERCLOS: {perclos*100:.1f}%",    (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        bar_w = int((self.alert_counter / CONSEC_FRAMES) * 200)
        cv2.rectangle(frame, (10, 130), (210, 145), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, 130), (10 + bar_w, 145), color, -1)
        if self.alarm_on:
            cv2.rectangle(frame, (0, h//2-40), (w, h//2+40), (0, 0, 180), -1)
            cv2.putText(frame, "DROWSY ALERT!", (w//2-160, h//2+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Starting — press ESC to quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res  = self.face_mesh.process(rgb)
            ear = mar = drowsy_prob = 0.0

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                ear = (eye_aspect_ratio(lm, LEFT_EYE,  w, h) +
                       eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2.0
                mar = mouth_aspect_ratio(lm, MOUTH, w, h)
                self.eye_states.append(1 if ear < EAR_THRESHOLD else 0)
                perclos = sum(self.eye_states) / len(self.eye_states)

                face_crop = crop_face(frame, lm, w, h)
                tensor    = TRANSFORM(Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)))
                self.frame_buffer.append(tensor)
                drowsy_prob = self.predict()

                is_drowsy = (
                    (drowsy_prob > MODEL_THRESHOLD) or
                    (ear < EAR_THRESHOLD and drowsy_prob > 0.4) or
                    (mar > MAR_THRESHOLD and drowsy_prob > 0.4) or
                    (perclos > 0.15 and drowsy_prob > 0.4)
                )

                if is_drowsy:
                    self.alert_counter = min(self.alert_counter + 1, CONSEC_FRAMES)
                    if self.alert_counter >= CONSEC_FRAMES:
                        self.trigger_alarm()
                else:
                    self.alert_counter = max(self.alert_counter - 1, 0)
                    if self.alert_counter == 0:
                        self.stop_alarm()
            else:
                perclos = sum(self.eye_states) / max(len(self.eye_states), 1)

            self.draw_hud(frame, ear, mar, drowsy_prob, perclos)
            cv2.imshow("Driver Monitor — ESC to quit", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.stop_alarm()
        pygame.mixer.quit()
        print("Session ended.")


if __name__ == "__main__":
    monitor = DrowsinessMonitor()
    monitor.run()