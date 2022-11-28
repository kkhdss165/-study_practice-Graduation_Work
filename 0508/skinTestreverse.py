import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../img/hyunbin.png')
img = cv2.imread('../img/sangsoo.png')
img = cv2.imread('../img/sukhun.png')
img = cv2.imread('../img/jungmin.png')
img = cv2.imread('../img/longhair.jpg')
image_file =['../img/hyunbin.png','../img/sangsoo.png','../img/sukhun.png','../img/jungmin.png','../img/longhair.jpg']
for i in image_file:
    img = cv2.imread(i)
    img_y, img_x, _ = img.shape
    face_img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    cv2.imshow('face_img_ycrcb', face_img_ycrcb)
    cv2.waitKey(0)



    lower = np.array([0,133,77], dtype = np.uint8)
    upper = np.array([255,173,127], dtype = np.uint8)

    # print(face_img_ycrcb)
    list = []
    for y in range(img_y):
        for x in range(img_x):
            if face_img_ycrcb[y][x][0]>=0 and face_img_ycrcb[y][x][0] <=255 and face_img_ycrcb[y][x][1]>=133 and face_img_ycrcb[y][x][0] <=173 and face_img_ycrcb[y][x][0]>=77 and face_img_ycrcb[y][x][0] <=127:
                print(face_img_ycrcb[y][x])
                list.append((y,x))

    new = np.zeros((img_y, img_x,3))
    for index in list:
        new[index[0],index[1]] =[0,0,255]

    cv2.imshow('new', new)
    cv2.waitKey(0)



    skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)

    print(skin_msk)

    skin = cv2.bitwise_and(img, img, mask = skin_msk)

    img_temp = skin.copy()
    img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = np.average(skin, axis=(0,1))

    cv2.imshow('img', skin)
    cv2.imshow('imssg', img_temp)
    cv2.imshow('skin_msk', skin_msk)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print(lower)