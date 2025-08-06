# Import the OpenCV library for computer vision tasks
import cv2

# Load the pre-trained Haar cascade for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained Haar cascade for detecting eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the pre-trained Haar cascade for detecting smiles
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Start capturing video from the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")  # Instruction for the user

# Main loop: run continuously until the user quits
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()  # ret = True if frame was read correctly, frame = image

    # If frame not read correctly, exit the loop
    if not ret:
        break

    # Convert the frame to grayscale (Haar cascades work better with gray images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # Returns a list of rectangles (x, y, width, height)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through all detected faces
    for (x, y, w, h) in faces:
        # Draw a blue rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Put the label "Face" above the rectangle
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Extract the region of interest (ROI) for face in both grayscale and color
        roi_gray = gray[y:y+h, x:x+w]     # Region for detecting eyes and smile (grayscale)
        roi_color = frame[y:y+h, x:x+w]   # Same region but in color for drawing rectangles

        # Detect eyes in the face region (smaller area = faster)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

        # Loop through detected eyes
        for (ex, ey, ew, eh) in eyes:
            # Draw a yellow rectangle around the eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)

            # Put the label "Eye" above the eye rectangle
            cv2.putText(roi_color, 'Eye', (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Detect smiles in the face region
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        # Loop through detected smiles
        for (sx, sy, sw, sh) in smiles:
            # Draw a green rectangle around the smile
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)

            # Put the label "Smile" above the smile rectangle
            cv2.putText(roi_color, 'Smile', (sx, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show the frame with all rectangles and labels in a window
    cv2.imshow('Face + Eyes + Smile Detector - Press q to Exit', frame)

    # Wait for 1 millisecond between frames, and check if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if user presses 'q'

# Release the webcam resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
