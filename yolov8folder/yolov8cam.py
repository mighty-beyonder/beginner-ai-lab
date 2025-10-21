from ultralytics import YOLO
import cv2

model = YOLO("E:/vs-code/virtual_environments/Plastic_and_organic_yolo_v8/plastic_detecting_robotic_arm/best.pt")
class_names = model.names

cap = cv2.VideoCapture(1)  

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (1020, 500))
    
    results = model.predict(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            cls_id = int(box.cls[0])
            label = class_names[cls_id]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('YOLO Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
