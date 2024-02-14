import cv2
from pathlib import Path
import torch
from ultralytics import YOLO
import math

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device loaded")
# Load YOLOv5 model
model = YOLO('best.pt')
print("model loaded")
# Initialize webcam
cap = cv2.VideoCapture(0)
print("webcam")
while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()
    # print(frame)
    if ret:
        # Perform inference
        results = model(frame)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                confidence = math.ceil((box.conf[0] * 100)) / 100
                if confidence>0.55:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    print(confidence)
                # confidence = math.ceil((box.conf[0] * 100)) / 100
                
                # text = f'test_tube {confidence}'
                # cv2.putText(frame, text, (max(0, x1), max(35, y1)), scale=1, thickness=2)
    #
        # Display output
        cv2.imshow('Test Tube Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
