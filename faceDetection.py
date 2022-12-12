import cv2
from random import randrange

# load some pre-trained data on face frontal fom opencv (Haarcascade Algorithm)
# trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# chose image to detect faces
img = cv2.imread('fake_ai_faces.png')

# Making it Grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# train algorithm

# detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

# draw rectangles around faces
for (x, y, w, h) in face_coordinates:
    # (x, y, w, h) = face_coordinates[1] for single face detection
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 3)
# putting direct values cv2.rectangle(img, (169, 130), (169+427, 130+427), (0, 255, 0), 2)

# print(face_coordinates)

# show image
cv2.imshow('Ak face Detector', img)
cv2.waitKey()


print('Code')

