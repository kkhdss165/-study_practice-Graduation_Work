import numpy as np
import cv2
import mediapipe as mp
import dlib

#face_mesh range list
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

#dlib rage list
RIGHT_EYEBROW_RANGE = list(range(17,22))   #17~21
LEFT_EYEBROW_RANGE = list(range(22,27))    #22~26

file_names =[ '../img/longhair.jpg', '../img/actor.jpg','../img/hyunbin.png', '../img/jungmin.png', '../img/sangsoo.png', '../img/sukhun.png',  '../img/longhair.jpg']

#remove background
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
    bit_wise_not = cv2.bitwise_not(output_image)
    new_res = bit_wise_not + img_bit

    cv2.imshow("image", image)
    cv2.imshow("output", output_image)
    cv2.imshow("img_bit", img_bit)
    cv2.imshow("new_res", new_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return new_res

#Kmeans
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
#printKmeans colors and shape convert
def showCenter(center):
    center = center.astype('int32')
    print("center_BGR:", center)
    pixel = np.uint8([center])

    center_hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    center_hsv = center[0]
    center_hsv = center.astype('int32')
    for i in range(len(center)):
        max_index = i
        for j in range(i + 1, len(center)):
            if center_hsv[max_index][0] + center_hsv[max_index][1] + center_hsv[max_index][2] < center_hsv[j][0] + center_hsv[j][1] + center_hsv[j][2]:
                max_index = j

        temp_color_hsv = center_hsv[i].copy()
        center_hsv[i] = center_hsv[max_index]
        center_hsv[max_index] = temp_color_hsv

        temp_color = center[i].copy()
        center[i] = center[max_index]
        center[max_index] = temp_color

    print("center_hsv:", center_hsv)

    center_image = np.full((100, 100 * len(center), 3), (0, 0, 0), np.uint8)
    for i in range(len(center)):
        center_image[:, (i * 100):(i + 1) * 100] = (center[i][0], center[i][1], center[i][2])
    cv2.imshow('center_image', center_image)
    cv2.waitKey()
    center = center.tolist()

    return center

#remove eyes and mouse area
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
            print(list)
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

#get howmany colors
def getFrequent(list):
    name=[]
    nums=[]
    for i in list:
        if i not in name:
            name.append(i)
            nums.append(1)
        else:
            index = name.index(i)
            nums[index] += 1

    max_index = nums.index(max(nums))
    return name[max_index]
def noise_oper(list, center):
    n_list = len(list)
    name = []
    nums = []

    for i in range(n_list):
        if list[i] not in name:
            name.append(list[i])
            nums.append(1)
        else:
            index = name.index(list[i])
            nums[index] += 1

    if center[0] in name:
        n_center_0 = nums[name.index(center[0])]
        if n_center_0 > n_list / 2:
            return center[0]

    sum_b = 0
    sum_g = 0
    sum_r = 0
    n_total = 0

    for i in range(len(name)):
        if name[i] not in [center[0]]:
            sum_b += name[i][0] * nums[i]
            sum_g += name[i][1] * nums[i]
            sum_r += name[i][2] * nums[i]

            n_total += nums[i]

    avg_color = [int(sum_b / n_total), int(sum_g / n_total), int(sum_r / n_total)]

    min_diff_color = None
    min_diff = None
    for i in range(len(name)):
        if name[i] not in [center[0]]:
            if min_diff == None:
                min_diff_color = name[i]
                min_diff = (avg_color[0] - name[i][0])**2 + (avg_color[1] - name[i][1])**2 + (avg_color[2] - name[i][2])**2

            else:
                now_diff = (avg_color[0] - name[i][0])**2 + (avg_color[1] - name[i][1])**2 + (avg_color[2] - name[i][2])**2
                if now_diff < min_diff :
                    min_diff_color = name[i]
                    min_diff = now_diff

    return min_diff_color
def removeNoise(image,center, size = 5):
    Total_group = []  # 컬러별로 픽셀들을 나누고 픽셀의 위치 리스트
    colors = []  # 컬러 순서 리스트
    img_y, img_x, _ = image.shape
    print("center[0]:", center[0])
    for i in range(0, int(img_y / size)):
        for j in range(0, int(img_x / size)):

            arround = []
            for idx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if ((i + idx[0]) * size >= 0 and (i + idx[0]) * size < img_y and (j + idx[1]) * size >= 0 and (j + idx[1]) * size < img_x):
                    arround.append(image[(i + idx[0]) * size][(j + idx[1]) * size].tolist())

            if image[i * size][j * size].tolist() not in arround:
                arround.append(image[idx[0] * size][idx[1] * size].tolist())
                image[i * size:(i + 1) * size, j * size:(j + 1) * size] =  noise_oper(arround, center)

            if image[i * size][j * size].tolist() not in colors:
                colors.append(image[i * size][j * size].tolist())
                Total_group.append([])

            index = colors.index(image[i * size][j * size].tolist())
            Total_group[index].append((i, j))

    cv2.imshow("img", image)
    print(colors)
    print(Total_group)
    cv2.waitKey(0)

    return image, Total_group, colors

def mosaic(image,size = 5):
    img_y, img_x, _ = image.shape
    for i in range(int(img_y/size)):
        for j in range(int(img_x/size)):
            arround =[]
            for x in range(size):
                for y in range(size):
                    arround.append(image[size*i+x][size*j+y].tolist())

            for x in range(size):
                for y in range(size):
                    image[size*i+x][size*j+y] = getFrequent(arround)

    print("image")
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.waitKey(0)

    return image
def sortbyLen(list):
    n_list = len(list)

    for i in range(n_list):
        min_index = i
        for j in range(i+1, n_list):
            if len(list[j]) < len(list[min_index]):
                min_index = j

        temp_space = list[min_index]
        list[min_index] = list[i]
        list[i] = temp_space

    return list

def createPixelGroup(image, Total_group, colors,center, size = 5):
    group_pixel_list = []  # 컬러 별로 나눈것을 거리에 따라 그룹화한 리스트
    img_y, img_x, _ = image.shape

    for i in range(len(colors)):

        color_group_list = []
        for index in Total_group[i]:
            if len(color_group_list) == 0:
                color_group_list.append([index])

            else:
                is_append = False
                for group_index in range(len(color_group_list)):
                    for around in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if (index[0] + around[0], index[1] + around[1]) in color_group_list[group_index]:
                            is_append = True
                            if index not in color_group_list[group_index]:
                                color_group_list[group_index].append(index)


                if is_append == False:
                    color_group_list.append([index])

        for part_index in range(len(color_group_list)):
            for search_index in range(part_index+1, len(color_group_list)):
                for index in color_group_list[part_index]:
                    for around in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if (index[0] + around[0], index[1] + around[1]) in color_group_list[search_index]:
                            for a_index in color_group_list[part_index]:
                                color_group_list[search_index].append(a_index)
                            color_group_list[part_index]=[]
                            break

        while [] in color_group_list:
            color_group_list.remove([])

        for part_list in color_group_list:
            group_pixel_list.append(part_list)

        # print("after sort , color_group_list", len(color_group_list), color_group_list)
        # # 시각화
        # temp2 = image.copy()
        # temp = np.zeros((img_y, img_x, 3), np.uint8)
        # for part_list in color_group_list:
        #     for i_index in part_list:
        #         temp2[i_index[0] * size:(i_index[0] + 1) * size, i_index[1] * size:(i_index[1] + 1) * size] = [0, 0,255]
        #     cv2.imshow("temp2", temp2)
        #     cv2.waitKey(0)

    return group_pixel_list

def groupingPixelGroup(image, group_pixel_list, center, size = 5):
    cv2.destroyAllWindows()
    cv2.imshow("before_image",  image)

    combine_list = sortbyLen(group_pixel_list)
    combine_image = image.copy()
    img_y, img_x, _ = image.shape
    img_max_x = int(img_x / size)
    img_max_y = int(img_y / size)

    excpetion_n = 0

    while len(combine_list) > 2 :
        for now_index in range(len(combine_list)):
            if len(combine_list[now_index]) > 0:
                now_index += excpetion_n
                break

        temp2 = image.copy()
        now_y = combine_list[now_index][0][0]*size
        now_x = combine_list[now_index][0][1]*size
        now_color = combine_image[now_y][now_x].tolist()
        near_group_list = []
        near_group_color = []
        near_pixel_list = []

        for index in combine_list[now_index]:
            for around in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                index_y = index[0] + around[0]
                index_x = index[1] + around[1]
                if (index_y, index_x) not in combine_list[now_index]:
                    if index_y in range(0,img_max_y) and index_x in range(0,img_max_x):
                        near_pixel_list.append((index_y,index_x))

        for index in near_pixel_list:
            for search_index in range(len(combine_list)):
                if index in combine_list[search_index]:
                    search_y = combine_list[search_index][0][0]
                    search_x = combine_list[search_index][0][1]
                    search_color = combine_image[search_y*size, search_x*size].tolist()
                    if search_color not in [center[0]] and search_index not in near_group_list:
                        near_group_list.append(search_index)
                        near_group_color.append(search_color)

        print("near_group_list", near_group_list)
        print("near_group_color",near_group_color)

        sum_b = 0
        sum_g = 0
        sum_r = 0
        n_total = 0
        for index in near_pixel_list + combine_list[now_index]:
            compare_color = combine_image[index[0]*size][index[1]*size].tolist()
            if compare_color not in [center[0]]:
                sum_b += compare_color[0]
                sum_g += compare_color[1]
                sum_r += compare_color[2]
                n_total += 1

        print("n_total", n_total)
        if len(near_group_list) > 1:
            avg_color = [int(sum_b / n_total), int(sum_g / n_total), int(sum_r / n_total)]
            diff_list = []
            diff_hsv_list =[]
            diff_sum = []
            for near_color in near_group_color:
                near_hsv = cv2.cvtColor(np.uint8([[near_color]]), cv2.COLOR_BGR2HSV)
                avg_hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)
                near_hsv = near_hsv[0][0]
                avg_hsv = avg_hsv[0][0]

                diff = (near_color[0] - avg_color[0]) ** 2 + (near_color[1] - avg_color[1]) ** 2 + (near_color[2] - avg_color[2]) ** 2 + 1
                diff_hsv = (near_hsv[0] - avg_hsv[0]) ** 2 + (near_hsv[1] - avg_hsv[1]) ** 2 + (near_hsv[2] - avg_hsv[2]) ** 2 + 1

                # if now_color == near_color:
                #     diff = -1
                diff_list.append(diff)
                diff_hsv_list.append(diff_hsv)

            avg = np.mean(diff_list)
            avg_hsv = np.mean(diff_hsv_list)

            std = np.std(diff_list)
            std_hsv = np.std(diff_hsv_list)



            diff_list = (diff_list/ std).tolist()
            diff_hsv_list = (diff_hsv_list / std_hsv).tolist()
            print("diff_list",diff_list)
            print("diff_hsv_list",diff_hsv_list)

            for n in range(len(near_group_color)):
                sum = 2/diff_list[n] + 1/diff_hsv_list[n]
                diff_sum.append(sum)

            min_index = diff_list.index(min(diff_list))
            min_index = diff_hsv_list.index(min(diff_hsv_list))
            min_index = diff_sum.index(max(diff_sum))
            combine_index = near_group_list[min_index]
            print("avg_color",avg_color)
            print("diff_list",diff_list)
            print("diff_hsv_list",diff_hsv_list)
            print("diff_sum", diff_sum)
            print("combine_index",min_index, combine_index)

            #시각화
            temp2 = np.zeros((img_y, img_x, 3), np.uint8)
            for index in combine_list[now_index]:
                temp2[index[0] * size:(index[0] + 1) * size, index[1] * size:(index[1] + 1) * size] = [0, 0, 255]

            for index in near_pixel_list:
                temp2[index[0] * size:(index[0] + 1) * size, index[1] * size:(index[1] + 1) * size] = combine_image[index[0] * size, index[1] * size].tolist()

            color_bar = np.zeros((150, 150 * (2 + len(near_group_list)), 3), np.uint8)
            color_bar[:150, :150] = now_color
            color_bar[:, 150:300] = avg_color
            for idx in range(len(near_group_color)):
                color_bar[:150, 150 * (idx + 2):150 * (idx + 3)] = near_group_color[idx]

            color_bar[145:150, 150 * (min_index + 2):150 * (min_index + 3)] = [0, 0, 255]

            if now_index != combine_index:
                for points in combine_list[now_index]:
                    # print(now_index , combine_index)
                    combine_list[combine_index].append(points)

                    combine_image[points[0] * size:(points[0] + 1) * size, points[1] * size:(points[1] + 1) * size] = near_group_color[min_index]

                combine_list[now_index] = []
                combine_list.remove([])

            cv2.imshow("colorbar", color_bar)
            cv2.imshow("temp2", temp2)
            cv2.imshow("combine_image", combine_image)
        else:
            excpetion_n += 1

        combine_list = sortbyLen(combine_list)
        cv2.waitKey(0)

def getHair(image,center, size = 5):
    # 이미지 노이즈 제거
    image, Total_group, colors = removeNoise(image,center, size)
    group_pixel_list = createPixelGroup(image, Total_group, colors, center, size)
    groupingPixelGroup(image, group_pixel_list, center, size)

if __name__ == "__main__":
    for file_name in file_names:
        image_orgin = cv2.imread(file_name)

        image = cv2.resize(image_orgin, (200,300))
        cv2.imshow("ss",image)
        cv2.waitKey(0)
        image = removeBack(image)

        K_image, center = kmeasImage(image, 15)
        center = showCenter(center)

        K_image, start_point = removeEyesandMouse(K_image)
        image = mosaic(K_image, 5)
        K_image = getHair(K_image, center, 5)

        cv2.destroyAllWindows()