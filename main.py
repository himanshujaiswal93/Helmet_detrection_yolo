from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(img, (x1, y1), (x2,y2), (255,0,255), 2)
            cv2.putText(img, f"{label} {conf:2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

            cv2.imshow("Webcam", img)
            cv2.waitKey(1)