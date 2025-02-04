import cv2
from random import randrange
print("CV2 version: " + cv2.__version__)

#load pre trained data from openCV (from the haarcascade file)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# image used
#img = cv2.imread('jim.jpg')

print("Face Detector, Press Q to quit")

webcam = cv2.VideoCapture(0)

while True:
    succesful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coords = trained_face_data.detectMultiScale(grayscaled_img)

    # have to use array as the face_coords is a double array as it can accoutn for multiple faces
    for (x, y, w, h) in face_coords:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 7)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    #Stop if Q key is pressed
    if key==81 or key==113:
        break

#Release the videoCapture
webcam.release()

#image code
"""
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coords = trained_face_data.detectMultiScale(grayscaled_img)

print(face_coords)

# have to use array as the face_coords is a double array as it can accoutn for multiple faces
for (x, y, w, h) in face_coords:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 7)



#display image with the faces
cv2.imshow('Clever Programmer Face Detector', img)
"""

cv2.waitKey()

print("Code Completed")