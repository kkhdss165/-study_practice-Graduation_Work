import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../img/hyunbin.png')
#img = cv2.imread('../img/sangsoo.png')
img = cv2.imread('../img/sukhun.png')
#img = cv2.imread('../img/jungmin.png')
#img = cv2.imread('../img/longhair.jpg')
face_img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# lower = np.array([0,133,77], dtype = np.uint8)
# upper = np.array([255,173,127], dtype = np.uint8)

lower = np.array([0,128,71], dtype = np.uint8)
upper = np.array([255,173,127], dtype = np.uint8)


skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)

print(skin_msk)

skin = cv2.bitwise_and(img, img, mask = skin_msk)

img_temp = skin.copy()
img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = np.average(skin, axis=(0,1))

cv2.imshow('img', skin)
cv2.imshow('imssg', img_temp)
cv2.waitKey()
cv2.destroyAllWindows()

print(lower)