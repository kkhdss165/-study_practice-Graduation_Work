import cv2
import numpy as np

image = np.zeros((150,150,3))

image[0, 0] =[0,255,0]

cv2.imshow("image", image)
cv2.waitKey(0)