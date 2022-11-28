from hair_detech_module import removeBack, hairsegement, post_processing
import cv2


ai = hairsegement()
rm = removeBack()
project_images = ['./img/shortcut.png','./img/actor.jpg', './img/longhair.jpg', './img/hyunbin.png', './img/jungmin.png', './img/sukhun.png']

def runSegement(file_name):
    mask = removeBack().getremovemask(file_name)
    overlay, prediction_colormap = hairsegement().gethairsegement(file_name)

    overlay = cv2.bitwise_and(overlay,overlay,mask=mask)
    prediction_colormap = cv2.bitwise_and(prediction_colormap,overlay,mask=mask)

    cv2.imshow("prediction_colormap", prediction_colormap)
    cv2.imshow("overlay", overlay)

    res, shift_image = post_processing(prediction_colormap).post_process()

    cv2.imshow("res", res)
    cv2.imshow("shift_image", shift_image)

    return overlay, res, shift_image

print("after call ai")
runSegement(project_images[0])
cv2.waitKey()
