import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Read the first frame and convert to gray
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

print("Motion detection started. Press 'q' to quit.")

while True:
    # Read next frame
    ret, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Calculate difference between frames
    diff = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours (moving objects)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Filter small movements
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame2, 'Motion Detected', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show live frame
    cv2.imshow('Motion Detector', frame2)

    # Exit on 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Set current frame as previous
    gray1 = gray2

cap.release()
cv2.destroyAllWindows()
