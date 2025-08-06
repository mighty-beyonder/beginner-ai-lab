import cv2
import numpy as np

# Open the webcam
cap = cv2.VideoCapture(0)

# Create a blank canvas to draw on
canvas = None

# Define HSV color range for blue object (tune this for your object)
lower_color = np.array([100, 150, 70])
upper_color = np.array([140, 255, 255])

# Store previous point to draw lines
prev_center = None

print("🖌️ Air Painter started! Use a blue object to draw. Press 'c' to clear. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Initialize canvas size
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the selected color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Remove noise from mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If a contour is found
    if contours:
        # Find largest contour
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 1000:
            # Get center of contour
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)

            # Draw a circle on detected object
            cv2.circle(frame, center, 10, (0, 255, 0), -1)

            # Draw on canvas if previous point exists
            if prev_center:
                cv2.line(canvas, prev_center, center, (255, 0, 0), 5)

            prev_center = center
        else:
            prev_center = None
    else:
        prev_center = None

    # Combine original frame with the drawing canvas
    output = cv2.add(frame, canvas)

    # Show result
    cv2.imshow("Air Painter", output)
    cv2.imshow("Color Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        # Clear canvas when 'c' is pressed
        canvas = np.zeros_like(frame)
        print("🧼 Canvas cleared!")

cap.release()
cv2.destroyAllWindows()
