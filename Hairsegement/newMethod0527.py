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

file_names =['../img/hyunbin.png', '../img/jungmin.png', '../img/sangsoo.png', '../img/sukhun.png', '../img/actor.jpg', '../img/longhair.jpg']

def removeBack(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    results = None
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    BG_COLOR = (0, 0, 0)  # gray
    MASK_COLOR = (255, 255, 255)  # white

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
    cv2.imshow("image", image)
    cv2.imshow("output", output_image)
    cv2.imshow("img_bit", img_bit)
    cv2.waitKey(0)
    return img_bit

def kmeasImage(image, K = 10):
    data = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 평균 클러스터링 적용 ---④
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # 중심 값을 정수형으로 변환 ---⑤
    center = np.uint8(center)
    #print(center)
    # 각 레이블에 해당하는 중심값으로 픽셀 값 선택 ---⑥
    res = center[label.flatten()]
    #print(label.flatten())
    # 원본 영상의 형태로 변환 ---⑦
    res = res.reshape((image.shape))
    # 결과 출력 ---⑧
    cv2.imshow('KMeans Color', res)
    cv2.waitKey(0)

    return res, center

def showCenter(center):
    center = center.astype('int32')
    pixel = np.uint8([center])

    center_hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    center_hsv = center[0]
    center_hsv = center.astype('int32')
    for i in range(len(center)):
        max_index = i
        for j in range(i + 1, len(center)):
            if center_hsv[max_index][0] + center_hsv[max_index][1] + center_hsv[max_index][2] < center_hsv[j][0] + \
                    center_hsv[j][1] + center_hsv[j][2]:
                max_index = j

        temp_color = center_hsv[i].copy()
        center_hsv[i] = center_hsv[max_index]
        center_hsv[max_index] = temp_color

        temp_color = center[i].copy()
        center[i] = center[max_index]
        center[max_index] = temp_color

    center_image = np.full((150, 150 * len(center), 3), (0, 0, 0), np.uint8)
    for i in range(len(center)):
        center_image[:, (i * 150):(i + 1) * 150] = (center[i][0], center[i][1], center[i][2])
    cv2.imshow('center_image', center_image)
    cv2.waitKey()
    center = center.tolist()

    return center

def removeEyesandMouse(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        result = face_mesh.process(rgb_image)

    img_y, img_x, _ = image.shape
    colors = []
    start_point=None

    for facial_landmarks in result.multi_face_landmarks:
        list = []
        for i in OUTLINE_POINTS_3:
            pt1 = facial_landmarks.landmark[i]
            list.append([int(pt1.x * img_x), int(pt1.y * img_y)])
        list = np.array(list, np.int32)

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [list], [255, 255, 255])
        cv2.imshow('maks', mask)
        bit_and = cv2.bitwise_and(mask, image)
        cv2.imshow('bit_and', bit_and)

        nums = []
        bit_and = bit_and.tolist()
        for i in range(img_y):
            for j in range(img_x):
                if bit_and[i][j] != [0, 0, 0]:
                    if bit_and[i][j] not in colors:
                        colors.append(bit_and[i][j])
                        nums.append(1)
                    else:
                        index = colors.index(bit_and[i][j])
                        nums[index] = nums[index] + 1

        for i in range(len(nums)):
            max_index = i
            for j in range(i + 1, len(nums)):
                if nums[max_index] < nums[j]:
                    max_index = j

            temp_nums = nums[i]
            temp_color = colors[i]

            nums[i] = nums[max_index]
            colors[i] = colors[max_index]

            nums[max_index] = temp_nums
            colors[max_index] = temp_color

        list = []
        for i in RIGHT_EYE_POINT_2:
            pt1 = facial_landmarks.landmark[i]
            list.append([int(pt1.x * img_x), int(pt1.y * img_y)])
        list = np.array(list, np.int32)
        cv2.fillPoly(image, [list], colors[0])

        list = []
        for i in LEFT_EYE_POINT_2:
            pt1 = facial_landmarks.landmark[i]
            list.append([int(pt1.x * img_x), int(pt1.y * img_y)])
        list = np.array(list, np.int32)
        cv2.fillPoly(image, [list], colors[0])

        list = []
        for i in MOUSE_POINT_5:
            pt1 = facial_landmarks.landmark[i]
            list.append([int(pt1.x * img_x), int(pt1.y * img_y)])
        list = np.array(list, np.int32)
        cv2.fillPoly(image, [list], colors[0])

        pt1 = facial_landmarks.landmark[9]
        start_point = (int(pt1.x * img_x), int(pt1.y * img_x))

    cv2.imshow('RemoveEyesandMouse', image)
    cv2.waitKey(0)

    return image, start_point

def getHairList(image, center, start_point):
    before = image[start_point[1]][start_point[0]]
    diff = []
    temp_copy = image.copy()

    for i in range(start_point[1], 0, -1):
        now = image[i][start_point[0]]
        now_index = center.index([now[0], now[1], now[2]])
        before_index = center.index([before[0], before[1], before[2]])

        now_image = np.full((150, 150, 3), (now[0], now[1], now[2]), np.uint8)
        before_image = np.full((150, 150, 3), (before[0], before[1], before[2]), np.uint8)

        print(now_index, before_index)

        if abs(now_index - before_index) > 1:
            cv2.line(temp_copy, [start_point[0], i], [start_point[0], i], (0, 0, 255), 2)
            diff.append(i)
            cv2.imshow('now', now_image)
            cv2.imshow('before', before_image)
            cv2.imshow('temp', image)
            cv2.imshow('temp_copy', temp_copy)
            cv2.waitKey(0)
        else:
            cv2.line(temp_copy, [start_point[0], i], [start_point[0], i], (255, 0, 0), 2)

        before = now

    return diff

def getHairArea(image, hair_list, start_point):
    hair_color_list = []
    img_y, img_x, _ = image.shape
    for i in hair_list:
        now = image[i][start_point[0]]
        now_index = center.index([now[0], now[1], now[2]])
        before = image[i + 1][start_point[0]]
        before_index = center.index([before[0], before[1], before[2]])

        if now_index > before_index:
            hair_color_list.append([now[0], now[1], now[2]])

    print(hair_color_list)

    for i in range(img_y):
        for j in range(img_x):
            now = image[i][j]
            if [now[0], now[1], now[2]] not in hair_color_list:
                image[i][j] = [255, 255, 255]

    cv2.imshow('temp', image)
    cv2.waitKey(0)
    cv2.waitKey(0)

if __name__ == "__main__":
    for file_name in file_names:
        image = cv2.imread(file_name)
        image = removeBack(image)
        image, center = kmeasImage(image)
        center = showCenter(center)
        image, start_point = removeEyesandMouse(image)

        hair_list = getHairList(image, center, start_point)

        getHairArea(image, hair_list)

        cv2.destroyAllWindows()