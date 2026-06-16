from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # pretrained model

model.train(
    data=r"C:\Users\prate\Desktop\safety ai\yolo converter\archive\data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="helmet_detector"
)