from hair_detech_1 import removeBack, hairsegement, post_processing
from hair_similarity_1 import similarity
import cv2


ai = hairsegement()
rm = removeBack()
sim = similarity()

file_name ='./img/hyunbin.png'

mask = removeBack().getremovemask(file_name)
overlay, prediction_colormap = hairsegement().gethairsegement(file_name)

overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
prediction_colormap = cv2.bitwise_and(prediction_colormap, overlay, mask=mask)

res = post_processing(prediction_colormap).post_process()

cv2.imshow('res', res)
cv2.waitKey()

similar_list = sim.get_similar_hair_list(res)
print(similar_list[:3])