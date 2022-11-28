import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

img = cv2.imread('./img/couple.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(gray)

print(len(faces))
for rect in faces:

    x, y = rect.left(), rect.top()
    w, h = rect.right() , rect.bottom()
    print(x,y,w,h)
    cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 1)


cv2.imshow('face detect', img)
cv2.waitKey()
cv2.destroyAllWindows()