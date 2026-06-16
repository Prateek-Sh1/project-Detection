from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\prate\Desktop\safety ai\yolo converter\archive\runs\detect\helmet_detector-2\weights\best.pt")

img = cv2.imread("your_test_image.jpg")

results = model(img, conf=0.25)
results[0].show()