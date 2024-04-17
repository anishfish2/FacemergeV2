import dlib
import numpy as np
import cv2

# Load the pre-trained face detector and landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You need to download this file

# Load the image
name = "anish"
desired_size = (256, 256)
image = cv2.imread(f"pngs/{name}.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = detector(gray)

# Predict facial landmarks
landmarks = predictor(gray, faces[0])

# Extract the coordinates of the left and right eye landmarks
left_eye = (landmarks.part(36).x, landmarks.part(36).y)
right_eye = (landmarks.part(45).x, landmarks.part(45).y)

# Calculate the angle of rotation
dY = right_eye[1] - left_eye[1]
dX = right_eye[0] - left_eye[0]
angle = np.degrees(np.arctan2(dY, dX))

# Calculate the center of rotation
center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

# Perform rotation
M = cv2.getRotationMatrix2D(center, angle, 1.0)
aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

# Resize the aligned face to the desired size
aligned_face_resized = cv2.resize(aligned_face, desired_size)
output_path = f"aligned/{name}.png"
cv2.imwrite(output_path, aligned_face)
