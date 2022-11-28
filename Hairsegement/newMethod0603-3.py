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

file_names =[ '../img/longhair.jpg', '../img/hyunbin.png', '../img/jungmin.png', '../img/sangsoo.png', '../img/sukhun.png', '../img/actor.jpg', '../img/longhair.jpg']

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

def groupingPixel(image, center, size = 5):
    Total_group =[]     #컬러별로 픽셀들을 나누고 픽셀의 위치 리스트
    colors=[]           #컬러 순서 리스트
    img_y, img_x, _ = image.shape
    print("center[0]:", center[0])
    for i in range(0,int(img_y/size)):
        for j in range(0,int(img_x/size)):

            arround = []
            for idx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    if ((i+idx[0])*size >= 0 and (i+idx[0])*size < img_y and (j+idx[1])*size >=0 and (j+idx[1])*size < img_x):
                        arround.append(image[(i+idx[0])*size][(j+idx[1])*size].tolist())

            if image[i*size][j*size].tolist() not in arround:
                image[i*size:(i+1)*size,j*size:(j+1)*size] = getFrequent(arround)

            if image[i*size][j*size].tolist() not in colors:
                colors.append(image[i*size][j*size].tolist())
                Total_group.append([])

            index = colors.index(image[i*size][j*size].tolist())
            Total_group[index].append((i,j))

    cv2.imshow("img", image)
    print(colors)
    print(Total_group)

    depart_group = [] #컬러 별로 나눈것을 거리에 따라 그룹화한 리스트

    for i in range(len(colors)):
        temp2 = image.copy()
        temp = np.zeros((img_y,img_x,3), np.uint8)
        for index in Total_group[i]:
            temp[index[0]*size:(index[0]+1)*size,index[1]*size:(index[1]+1)*size] = [255,255,255]

        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        ret, imthres = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        imthres = cv2.bitwise_not(imthres)

        contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

        for a in contour:
            for b in a:
                cv2.circle(temp2, tuple(b[0]), 1, (255, 0, 0), 1)
            group = np.zeros((img_y,img_x,3), np.uint8)
            cv2.fillPoly(group, [a[:,0]],[255,255,255])
            # print([a[:,0]])
            group_list =[]
            for y in range(0, int(img_y / size)):
                for x in range(0, int(img_x / size)):
                    # print(group[y*size + int(size/2)][x*size + int(size/2)].tolist())
                    if group[y*size + int(size/2)][x*size + int(size/2)].tolist() in [[255,255,255]] :
                        group_list.append((y,x))
            # print("group_list : ",group_list)


            cv2.imshow("group", group)
            cv2.imshow("gray", gray)
            cv2.imshow("temp2", temp2)

            #show red screen
            # group_test = np.zeros((img_y,img_x,3), np.uint8)
            # for index in group_list:
            #     group_test[index[0]*size:(index[0]+1)*size,index[1]*size:(index[1]+1)*size] = [0,0,255]
            # cv2.imshow("group_test", group_test)

            depart_group.append(group_list)

            # cv2.waitKey(0)

        cv2.imshow("gray", gray)
        # cv2.waitKey(0)

    print("len: ",len(depart_group))
    print(depart_group)
    # ##show red screen
    # for points in depart_group:
    #     group_test = np.zeros((img_y, img_x, 3), np.uint8)
    #     for point in points:
    #         group_test[point[0]*size:(point[0]+1)*size,point[1]*size:(point[1]+1)*size] = [0,0,255]
    #     cv2.imshow("group_test", group_test)
    #     cv2.waitKey(0)

    near_list =[]
    for group_index in range(len(depart_group)):
        near = []
        for index in depart_group[group_index]:
            for around in [(-1,0),(1,0),(0,-1),(0,1)]:
                for search_index in range(len(depart_group)):
                    if (index[0]+around[0], index[1]+around[1]) in depart_group[search_index]:
                        if group_index != search_index:
                            if search_index not in near:
                                near.append(search_index)

        near_list.append(near)
    print("near_list",near_list)
    #     #인접한 컬러 그룹 시각화
    #     group_test = np.zeros((img_y, img_x, 3), np.uint8)
    #
    #     for near_group_index in near:
    #         for near_point in depart_group[near_group_index]:
    #             group_test[near_point[0] * size:(near_point[0] + 1) * size, near_point[1] * size:(near_point[1] + 1) * size] = image[near_point[0]*size + int(size/2)][near_point[1]*size + int(size/2)]
    #
    #     for point in depart_group[group_index]:
    #         group_test[point[0] * size:(point[0] + 1) * size, point[1] * size:(point[1] + 1) * size] = [0,255,0]
    #
    #     print(group_index,near)
    #     cv2.imshow("near", group_test)
    #     # cv2.waitKey(0)
    #
    #
    # cv2.imshow("temp",temp)
    # cv2.waitKey(0)


    pixel_nums=[] #현재 인덱스 그룹의 픽셀의 갯수

    for list_index in range(len(depart_group)):
        print()
        print(list_index,len(depart_group[list_index]),depart_group[list_index])
        pixel_nums.append((list_index, len(depart_group[list_index])))

    print("before sort:",pixel_nums)
    for i in range(len(pixel_nums)):
        min_index = i
        for j in range(i+1, len(pixel_nums)):
            if (pixel_nums[min_index][1] > pixel_nums[j][1]):
                min_index = j

        temp_1 = pixel_nums[i]
        pixel_nums[i] = pixel_nums[min_index]
        pixel_nums[min_index] = temp_1

    print("after sort:",pixel_nums)

    after_combine_list = depart_group.copy()
    after_combine_image = image.copy()

    cv2.destroyAllWindows()
    cv2.imshow("before_combine_image",after_combine_image)
    cv2.waitKey(0)

    for i in range(len(pixel_nums)):
        now_group_index = pixel_nums[i][0]
        near_index = []
        lens =[]
        near_colors =[]
        adjoin_pixel_nums =[]

        now_y = depart_group[now_group_index][0][0] * size
        now_x = depart_group[now_group_index][0][1] * size

        now_color = after_combine_image[now_y][now_x]

        for now_near_index in near_list[now_group_index]:

            near_y = depart_group[now_near_index][1][0] * size
            near_x = depart_group[now_near_index][1][1] * size

            print("now:" ,now_group_index, now_color,len(depart_group[now_group_index]),
                  "near:",now_near_index,after_combine_image[near_y][near_x],len(after_combine_list[now_near_index]))

            # 조건
            if after_combine_image[near_y][near_x].tolist() not in [center[0]] and after_combine_image[now_y][now_x].tolist() not in [center[0]]:
                if len(after_combine_list[now_near_index]) >= len(depart_group[now_group_index]):
                # print(len(after_combine_list[now_near_index]),len(depart_group[now_group_index]))
                    near_index.append(now_near_index)
                    lens.append(len(after_combine_list[now_near_index]))
                    near_colors.append(after_combine_image[near_y][near_x].tolist())

        #현재 그룹에 직접 접해있는 그룹별 픽셀의 수
        for j in range(len(near_index)):
            adjoin_pixel_nums.append(0)

        for index in depart_group[now_group_index]:
            for around in [(-1,0),(1,0),(0,-1),(0,1)]:
                for n_index in near_index:
                    # print(n_index, index)
                    if(index[0] + around[0], index[1] + around[1]) in depart_group[n_index]:
                        adjoin_pixel_nums[near_index.index(n_index)] += 1

        avg_color =[]
        now_color = now_color.tolist()
        for idx in range(len(near_index)):
            compare_color = near_colors[idx]
            len_now = len(depart_group[now_group_index])
            len_compare = lens[idx]
            sum_b = (compare_color[0] * len_compare + now_color[0] * len_now) / (len_compare + len_now)
            sum_g = (compare_color[1] * len_compare + now_color[1] * len_now) / (len_compare + len_now)
            sum_r = (compare_color[2] * len_compare + now_color[2] * len_now) / (len_compare + len_now)
            avg_color.append((int(sum_b), int(sum_g), int(sum_r)))

        print(near_colors)
        print(avg_color)

        # print("adjoin_pixel_nums",adjoin_pixel_nums)
        # 가중치 계산
        # print(near_index)
        print(lens)
        if len(near_index) != 0:
            compare_list =[]

            for idx in range(len(near_index)):

                sum = (avg_color[idx][0] - near_colors[idx][0])**2 + (avg_color[idx][1] - near_colors[idx][1])**2 + (avg_color[idx][2] - near_colors[idx] [2])**2
                print(idx,sum)

                compare_list.append(sum)
            print(compare_list)
            color_index = compare_list.index(min (compare_list))

            if now_color in near_colors:
                color_index = near_colors.index(now_color)

            combine_index = near_index[color_index]
            # combine_index = near_index[lens.index(min(lens))]
            # color_index = lens.index(min(lens))
            # combine_index = near_index[adjoin_pixel_nums.index(max(adjoin_pixel_nums))]
            # color_index = adjoin_pixel_nums.index(max(adjoin_pixel_nums))
            print(combine_index)

            #현재 픽셀, 평균 픽셀, 주변 픽셀의 시각화
            color_bar = np.zeros((300,150*(1+len(near_index)),3),np.uint8)
            color_bar[:150,:150] = now_color
            for idx in range(len(near_index)):
                color_bar[:150,150*(idx+1):150*(idx+2)] = near_colors[idx]
            for idx in range(len(near_index)):
                color_bar[150:300,150*(idx+1):150*(idx+2)] = avg_color[idx]

            color_bar[149:151, 150*(color_index+1):150*(color_index+2)] = [0,0,255]

            cv2.imshow("colorbar", color_bar)

            process_image = after_combine_image.copy()
            for point in after_combine_list[now_group_index]:
                process_image[point[0] * size:(point[0] + 1) * size, point[1] * size:(point[1] + 1) * size] = [0, 255, 0]
            cv2.imshow("process_image", process_image)

            # print(after_combine_list[combine_index], after_combine_list[now_group_index])

            #변경후 업데이트
            # print(after_combine_list[now_group_index])
            if now_group_index != combine_index:
                for points in after_combine_list[now_group_index]:
                    # print(now_group_index, combine_index)
                    after_combine_list[combine_index].append(points)

                    after_combine_image[points[0]*size:(points[0]+1)*size,points[1]*size:(points[1]+1)*size] = near_colors[color_index]


                after_combine_list[now_group_index] =[]

                for idx in near_list[now_group_index]:
                    if now_group_index in near_list[idx]:
                        print(idx,near_list[idx])
                        near_list[idx].remove(now_group_index)
                        print(idx,near_list[idx])
                        for n_idx in near_list[now_group_index]:
                            if n_idx not in near_list[idx] and n_idx != idx:
                                if n_idx == combine_index:
                                    near_list[idx].append(n_idx)
                                else:
                                    if combine_index not in near_list[idx]:
                                        near_list[idx].append(combine_index)


                        print(idx, near_list[idx])
                    # for near_i in near_index:
                    #     if near_i not in near_list[idx] and near_i != idx:
                    #         near_list[idx].append((near_i))

                near_list[now_group_index] = []

                # print(after_combine_list[combine_index], after_combine_list[now_group_index])

        cv2.imshow("after_combine_image", after_combine_image)
        cv2.waitKey(0)

    cv2.imshow("final" , after_combine_image)
    cv2.waitKey(0)

    return image

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

if __name__ == "__main__":
    for file_name in file_names:
        image_orgin = cv2.imread(file_name)

        image = cv2.resize(image_orgin, (200,300))
        cv2.imshow("ss",image)
        cv2.waitKey(0)
        image = removeBack(image)

        K_image, center = kmeasImage(image, 15 )
        center = showCenter(center)

        K_image, start_point = removeEyesandMouse(K_image)
        image = mosaic(K_image, 5)
        K_image = groupingPixel(K_image, center, 5)

        cv2.destroyAllWindows()