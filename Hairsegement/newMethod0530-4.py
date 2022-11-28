import numpy as np
import cv2
import mediapipe as mp
import dlib

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

NOSTRILL_POINT = [79,166,75,60,20,238, 309,392,305,290,250,458,459]

RIGHT_EYEBROW_RANGE = list(range(17,22))   #17~21
LEFT_EYEBROW_RANGE = list(range(22,27))    #22~26

file_names =[ '../img/longhair.jpg', '../img/hyunbin.png', '../img/jungmin.png', '../img/sangsoo.png', '../img/sukhun.png', '../img/actor.jpg', '../img/longhair.jpg']

def removeBack(image):
    img_y, img_x, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    results = None
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    BG_COLOR = (0, 0, 0)  # gray
    BG_COLOR_2 = (0, 255, 0)  # gray
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
    bit_wise_not = cv2.bitwise_not(output_image)
    new_res = bit_wise_not + img_bit

    cv2.imshow("image", image)
    cv2.imshow("output", output_image)
    cv2.imshow("img_bit", img_bit)
    cv2.waitKey(0)

    for i in range(img_y):
        for j in range(img_x):
            # print(output_image[i][j].tolist() )
            if output_image[i][j].tolist() == [0,0,0]:
                img_bit[i][j] = [0,255,0]

    # cv2.imshow("new_res", new_res)
    cv2.imshow("new_res2", img_bit)
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

def removeEyesandMouse(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

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

        for rect in faces:
            shape = predictor(gray, rect)
            list = []
            for i in range(9):
                pt1 = facial_landmarks.landmark[RIGHT_EYE_POINT_2[i]]
                list.append([int(pt1.x * img_x), int(pt1.y * img_y)])
            for i in reversed(RIGHT_EYEBROW_RANGE):
                part = shape.part(i)
                list.append((part.x, part.y))

            list = np.array(list, np.int32)
            cv2.fillPoly(image, [list], colors[0])

            list = []
            for i in range(9):
                pt1 = facial_landmarks.landmark[LEFT_EYE_POINT_2[i]]
                list.append([int(pt1.x * img_x), int(pt1.y * img_y)])

            for i in reversed(LEFT_EYEBROW_RANGE):
                part = shape.part(i)
                list.append((part.x, part.y))

            list = np.array(list, np.int32)
            cv2.fillPoly(image, [list], colors[0])

        list = []
        for i in MOUSE_POINT_5:
            pt1 = facial_landmarks.landmark[i]
            list.append([int(pt1.x * img_x), int(pt1.y * img_y)])
        list = np.array(list, np.int32)
        cv2.fillPoly(image, [list], colors[0])

        list = []
        for i in range(int(len(NOSTRILL_POINT)/2)):
            pt1 = facial_landmarks.landmark[NOSTRILL_POINT[i]]
            list.append([int(pt1.x * img_x), int(pt1.y * img_y)])
        list = np.array(list, np.int32)
        cv2.fillPoly(image, [list], colors[0])

        list = []
        for i in range(int(len(NOSTRILL_POINT)/2), len(NOSTRILL_POINT)):
            pt1 = facial_landmarks.landmark[NOSTRILL_POINT[i]]
            list.append([int(pt1.x * img_x), int(pt1.y * img_y)])
        list = np.array(list, np.int32)
        cv2.fillPoly(image, [list], colors[0])

        pt1 = facial_landmarks.landmark[9]
        start_point = (int(pt1.x * img_x), int(pt1.y * img_x))

    cv2.imshow('RemoveEyesandMouse', image)
    cv2.waitKey(0)

    return image, start_point


def grabcut(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    x,y,w,h = None,None,None,None

    for rect in faces:
        x, y = rect.left(), rect.top()
        w, h = rect.right(),rect.bottom()

        shape = predictor(gray, rect)
        part = shape.part(57)
        h = part.y

    temp = image.copy()
    roi = temp[y:h, x:w]
    cv2.rectangle(temp, (x,y), (w, h),(0,255,0),1)

    cv2.imshow("temp",temp)
    cv2.waitKey(0)

    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist_roi = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])

    bp = cv2.calcBackProject([hsv_img], [0, 1], hist_roi, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(bp, -1, disc, bp)
    _, mask = cv2.threshold(bp, 1, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(temp, temp, mask=mask)
    cv2.imshow("resutl", result)
    cv2.waitKey(0)

def skinTest(image):
    face_img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)
    skin_msk = cv2.bitwise_not(skin_msk)
    cv2.imshow('skin_mask', skin_msk)

    skin = cv2.bitwise_and(image, image, mask=skin_msk)

    img_temp = skin.copy()
    img_temp[:, :, 0], img_temp[:, :, 1], img_temp[:, :, 2] = np.average(skin, axis=(0, 1))

    cv2.imshow('img', skin)
    cv2.imshow('imssg', img_temp)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    for file_name in file_names:
        image_orgin = cv2.imread(file_name)

        image = removeBack(image_orgin)
        # image,_=kmeasImage(image)
        skinTest(image)
