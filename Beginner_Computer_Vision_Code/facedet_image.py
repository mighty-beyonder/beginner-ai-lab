# pip install opencv-python
import cv2

# ✅ Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier('E:/haarcascade/haarcascade_frontalface_default.xml')

# ✅ Load image file
image_path = r"E:/vs-code/virtual_environments/teaching_AI_ML_DL_CV/teaching/face.jpg"  # ← your image path
img = cv2.imread(image_path)

if img is None:
    print("Could not load image. Check the path.")
    exit()

# ✅ Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ✅ Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# ✅ Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# ✅ Display the output
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
