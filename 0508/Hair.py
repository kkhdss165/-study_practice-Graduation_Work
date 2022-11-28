import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

img = cv2.imread('../img/hyunbin.png')
#img = cv2.imread('../img/sangsoo.png')
#img = cv2.imread('../img/sukhun.png')
#img = cv2.imread('../img/jungmin.png')

face_img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

lower = np.array([0,133,77], dtype = np.uint8)
upper = np.array([255,173,127], dtype = np.uint8)


skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)

cv2.imshow('skin_msk', skin_msk)
cv2.waitKey()

skin = cv2.bitwise_and(img, img, mask = skin_msk)

img_temp = skin.copy()
img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = np.average(skin, axis=(0,1))



cv2.imshow('img', skin)
cv2.imshow('imssg', img_temp)
cv2.waitKey()
cv2.destroyAllWindows()


mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# For static images:
BG_COLOR = (192, 192, 192) # gray
BG_COLOR = (0,0,0) # gray
MASK_COLOR = (255, 255, 255) # white
with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
    image_height, image_width, _ = img.shape
    # Convert the BGR image to RGB before processing.
    results = selfie_segmentation.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    # Generate solid color images for showing the output selfie segmentation mask.
    fg_image = np.zeros(img.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(img.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    output_image = np.where(condition, fg_image, bg_image)
    img_bit = cv2.bitwise_and(img, output_image)
    cv2.imshow("output",output_image)
    cv2.imshow("img_bit", img_bit)

    print(skin_msk.size)
    print(output_image.size)
    gray_output = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

    result = cv2.bitwise_and(skin_msk, gray_output)
    cv2.imshow('skin_msk', skin_msk)
    cv2.imshow("result", result)
    cv2.waitKey(0)

    result2 = cv2.bitwise_xor(result, gray_output)

    cv2.imshow("result_2", result2)
    cv2.waitKey(0)
