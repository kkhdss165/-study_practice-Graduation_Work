import cv2

img = cv2.imread('../img/longhair.jpg')
img = cv2.resize(img, (200,300))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for i in range(1,240):
    ret, dst = cv2.threshold(gray, i, 255, cv2.THRESH_BINARY)
    cv2.imshow("gray", dst)
    cv2.waitKey(0)