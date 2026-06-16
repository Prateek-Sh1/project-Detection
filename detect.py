from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # lightweight model

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    results = model(frame)
    for r in results:
        for box in r.boxes:
            cls=int(box.cls[0])
            label=model.names[cls]

            if label=="person":
                cv2.putText(frame,"Alert: No Helmet!",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,225),3)

    annotated_frame = results[0].plot()

    cv2.imshow("Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()