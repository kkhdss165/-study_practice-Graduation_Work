import numpy as np
import cv2
import mediapipe as mp

OUTLINE_POINTS_1 = [10,109,67,103,54,21,162,127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338]    #36개
OUTLINE_POINTS_2 = [151,108,69,104,68,71,139,34,227,137,177,215,138,135,169,170,140,171,175,396,369,395,394,364,367,435,401,366,447,264,368,301,298,333,299,337]  #36개
OUTLINE_POINTS_3 = [9,107,66,105,63,70,156,143,116,123,147,213,192,214,210,211,32,208,199,428,262,431,430,434,416,433,376,352,345,372,383,300,293,334,296,336]    #36개
OUTLINE_POINTS_4 = [8,55,65,52,53,46,124,35,111,117,50,187,207,216,212,202,204,194,201,200,421,418,424,422,432,436,427,411,280,346,340,265,353,276,283,282,295,285] #38개

RIGHT_EYE_POINT_1 = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]   #16개
RIGHT_EYE_POINT_2 = [130,25,110,24,23,22,26,112,243,190,56,28,27,29,30,247]   #16개
RIGHT_EYE_POINT_3 = [226,31,228,229,230,231,232,233,244,189,221,222,223,224,225,113]   #16개

LEFT_EYE_POINT_1 = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]   #16개
LEFT_EYE_POINT_2 = [463,341,256,252,253,254,339,255,359,467,260,259,257,258,286,414]   #16개
LEFT_EYE_POINT_3 = [464,453,452,451,450,449,448,261,446,342,445,444,443,442,441,413]   #16개

MOUSE_POINT_1 = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]  #20개
MOUSE_POINT_2 = [62,96,89,179,86,15,316,403,319,325,292,407,272,271,268,12,38,41,42,183]  #20개
MOUSE_POINT_3 = [76,77,90,180,85,16,315,404,320,307,306,408,304,303,302,11,72,73,74,184]  #20개
MOUSE_POINT_4 = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]  #20개
MOUSE_POINT_5 = [57,43,106,182,83,18,313,406,335,273,287,410,322,391,393,164,167,165,92,186] #20개

# image = cv2.imread('../img/hyunbin.png')
# image = cv2.imread('../img/jungmin.png')
# image = cv2.imread('../img/sangsoo.png')
image = cv2.imread('../img/sukhun.png')
# image = cv2.imread('../img/actor.jpg')
image = cv2.imread('../img/longhair.jpg')
img_y, img_x, _ = image.shape

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation
result = None
results = None
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
    result = face_mesh.process(rgb_image)

BG_COLOR = (192, 192, 192) # gray
BG_COLOR = (0,0,0) # gray
MASK_COLOR = (255, 255, 255) # white

with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
    results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
# Generate solid color images for showing the output selfie segmentation mask.
fg_image = np.zeros(image.shape, dtype=np.uint8)
fg_image[:] = MASK_COLOR
bg_image = np.zeros(image.shape, dtype=np.uint8)
bg_image[:] = BG_COLOR
output_image = np.where(condition, fg_image, bg_image)
img_bit = cv2.bitwise_and(image, output_image)
cv2.imshow("output",output_image)
cv2.imshow("img_bit", img_bit)
cv2.waitKey(0)


temp = img_bit

for facial_landmarks in result.multi_face_landmarks:

    pt1 = facial_landmarks.landmark[9]
    start_point = (int(pt1.x * img_x),int(pt1.y * img_x))



cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('temp', temp)
cv2.waitKey(0)

# mostcolor = mostcolor[:7]
# spc = temp[start_point[0]][start_point[1]]
# print(spc)
# print(mostcolor)
#
# for i in range(start_point[1],0,-1):
#     now = temp[start_point[0]][i]
#     if [now[0], now[1], now[2]] != [spc[0], spc[1], spc[2]] and [now[0], now[1], now[2]] not in mostcolor:
#         cv2.line(temp, [start_point[0] , i], [start_point[0] , i], (0, 0, 255), 1)
image_hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
image_hsv = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
image_hsv = image_hsv.astype('int32')
before = image_hsv[start_point[1]][start_point[0]]
diff = []
print(img_x,img_y)

temp_copy = temp.copy()
cv2.line(temp_copy, start_point, start_point, (255,0,0), 1)
for i in range(start_point[1],0,-1):
    now = image_hsv[i][start_point[0]]

    loss_sum = ((now[0]-before[0])/2)**2++((now[1]-before[1])*2)**2+((now[2]-before[2])*2)**2
    print(before, now , (now[0]-before[0]), (now[1]-before[1]), (now[2]-before[2]),loss_sum)

    now_image =  np.full((200,200,3), (now[0],now[1],now[2]), np.uint8)
    before_image = np.full((200,200,3), (before[0],before[1], before[2]), np.uint8)

    if loss_sum > 2000:
        cv2.line(temp_copy , [start_point[0], i], [start_point[0], i], (0, 0, 255), 2)
        diff.append((i,loss_sum))
    else:
        cv2.line(temp_copy, [start_point[0], i], [start_point[0], i], (255, 0, 0), 2)
    cv2.imshow('now', now_image)
    cv2.imshow('before', before_image)
    cv2.imshow('temp', temp)
    cv2.imshow('temp_copy', temp_copy)
    cv2.waitKey(0)
    before = now

print(diff)
max_index = diff.index(max(diff))
print(max_index)


cv2.imshow('temp', temp)
cv2.waitKey(0)