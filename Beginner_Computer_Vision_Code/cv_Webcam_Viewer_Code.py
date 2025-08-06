import cv2

# Open the default webcam (0 = default camera)
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was not read correctly, exit
    if not ret:
        break

    # Display the frame in a window
    cv2.imshow("Live Camera - Press 'q' to Exit", frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
