import streamlit as st
import cv2
import time
from ultralytics import YOLO
import threading

class VideoCaptureThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.ret, self.frame = self.cap.read()
        self.running = True

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()




st.title("🚀 Smart Safety Monitoring System")
st.sidebar.title("📊 System Info")
st.sidebar.write("Model: YOLOv8")
st.sidebar.write("Status: Running")

model = YOLO(r"C:\Users\prate\Desktop\safety ai\yolo converter\archive\runs\detect\helmet_detector-2\weights\best.pt")

# Controls
start = st.button("Start Camera")
stop = st.button("Stop Camera")

frame_window = st.image([])

if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True

if stop:
    st.session_state.run = False


cap = VideoCaptureThread(0).start()

count = 0
last_alert_time = 0
cooldown = 3 
frame_count=0

while st.session_state.run:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera not working")
        break

    
    frame_count += 1

    if frame_count % 3 != 0:
        frame_window.image(frame, channels="BGR")
        continue

    results = model(frame,imgsz=320)
    current_time = time.time()

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            print("Detected:", label)
            
            if label == "no_helmet":
                if current_time - last_alert_time > cooldown:
                    count += 1
                    last_alert_time = current_time

                cv2.putText(frame, "⚠️ ALERT: No Helmet!", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                    
    st.sidebar.write("Violations:", count)
    frame_window.image(frame, channels="BGR")

cap.stop()