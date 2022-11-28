import cv2
import mediapipe as mp
import glob


OUTLINE_POINTS_1 = [10,109,67,103,54,21,162,127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338]    #36개
OUTLINE_POINTS_2 = [151,108,69,104,68,71,139,34,227,137,177,215,138,135,169,170,140,171,175,396,369,395,394,364,367,435,401,366,447,264,368,301,298,333,299,337]  #36개
OUTLINE_POINTS_3 = [9,107,66,105,63,70,156,143,116,123,147,213,192,214,210,211,32,208,199,428,262,431,430,434,416,433,376,352,345,372,383,300,293,334,296,336]    #36개
OUTLINE_POINTS_4 = [8,55,65,52,53,46,124,35,111,117,50,187,207,216,212,202,204,194,201,200,421,418,424,422,432,436,427,411,280,346,340,265,353,276,283,282,295,285] #38개

RIGHT_OUTLINE_POINT = [128,121,120,119,118,101,100,47,114,205,36,206,203]
LEFT_OUTLINE_POINT = [357,350,349,348,347,330,329,277,343,425,266,426,423]

#눈
RIGHT_EYE_POINT_1 = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]   #16개
RIGHT_EYE_POINT_2 = [130,25,110,24,23,22,26,112,243,190,56,28,27,29,30,247]   #16개
RIGHT_EYE_POINT_3 = [226,31,228,229,230,231,232,233,244,189,221,222,223,224,225,113]   #16개

LEFT_EYE_POINT_1 = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]   #16개
LEFT_EYE_POINT_2 = [463,341,256,252,253,254,339,255,359,467,260,259,257,258,286,414]   #16개
LEFT_EYE_POINT_3 = [464,453,452,451,450,449,448,261,446,342,445,444,443,442,441,413]   #16개

#입술
MOUSE_POINT_1 = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]  #20개
MOUSE_POINT_2 = [62,96,89,179,86,15,316,403,319,325,292,407,272,271,268,12,38,41,42,183]  #20개
MOUSE_POINT_3 = [76,77,90,180,85,16,315,404,320,307,306,408,304,303,302,11,72,73,74,184]  #20개
MOUSE_POINT_4 = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]  #20개
MOUSE_POINT_5 = [57,43,106,182,83,18,313,406,335,273,287,410,322,391,393,164,167,165,92,186] #20개

NOSE_POINT_1 = [168,193,245,188,174,217,126,142,129,98,97,2,326,327,358,371,355,437,399,412,465,417]
MIDDLE_NOSE_LINE = [6,197,195,5,4,1,19,94]
RIGHT_NOSE_POINT = [122,196,3,51,45,44,125,141, 236,134,220,237,241,242, 198,131,115,218,239,238,20,79, 209,49,48,219,166,60,99, 75,240,59,235,64,102]
LEFT_NOSE_POINT = [351,419,248,281,275,274,354,370, 456,363,440,457,461,462, 420,360,344,438,459,458,250,309, 429,279,278,439,392,290,328, 305,460,289,455,294,331]

RIGHT_IRIS = [468,469,470,471,472]
LEFT_IRIS = [473,474,475,476,477]

middle_index = []
right_index = []
left_index = []

def relocation_list():

    for i in range(int(len(OUTLINE_POINTS_1) / 2)):
        if i == 0:
            middle_index.append(OUTLINE_POINTS_1[i])
            middle_index.append(OUTLINE_POINTS_1[i + int(len(OUTLINE_POINTS_1) / 2)])
        else:
            right_index.append(OUTLINE_POINTS_1[i])
            left_index.append(OUTLINE_POINTS_1[len(OUTLINE_POINTS_1) - i])

    for i in range(int(len(OUTLINE_POINTS_2) / 2)):
        if i == 0:
            middle_index.append(OUTLINE_POINTS_2[i])
            middle_index.append(OUTLINE_POINTS_2[i + int(len(OUTLINE_POINTS_2) / 2)])
        else:
            right_index.append(OUTLINE_POINTS_2[i])
            left_index.append(OUTLINE_POINTS_2[len(OUTLINE_POINTS_2) - i])

    for i in range(int(len(OUTLINE_POINTS_3) / 2)):
        if i == 0:
            middle_index.append(OUTLINE_POINTS_3[i])
            middle_index.append(OUTLINE_POINTS_3[i + int(len(OUTLINE_POINTS_3) / 2)])
        else:
            right_index.append(OUTLINE_POINTS_3[i])
            left_index.append(OUTLINE_POINTS_3[len(OUTLINE_POINTS_3) - i])

    for i in range(int(len(OUTLINE_POINTS_4) / 2)):
        if i == 0:
            middle_index.append(OUTLINE_POINTS_4[i])
            middle_index.append(OUTLINE_POINTS_4[i + int(len(OUTLINE_POINTS_4) / 2)])
        else:
            right_index.append(OUTLINE_POINTS_4[i])
            left_index.append(OUTLINE_POINTS_4[len(OUTLINE_POINTS_4) - i])

    for i in RIGHT_OUTLINE_POINT:
        right_index.append(i)

    for i in RIGHT_EYE_POINT_1:
        right_index.append(i)

    for i in RIGHT_EYE_POINT_2:
        right_index.append(i)

    for i in RIGHT_EYE_POINT_3:
        right_index.append(i)

    for i in LEFT_OUTLINE_POINT:
        left_index.append(i)

    for i in range(len(LEFT_EYE_POINT_1)):
        if i <= 8:
            left_index.append(LEFT_EYE_POINT_1[8 - i])
        else:
            left_index.append(LEFT_EYE_POINT_1[24 - i])

    for i in range(len(LEFT_EYE_POINT_2)):
        if i <= 8:
            left_index.append(LEFT_EYE_POINT_2[8 - i])
        else:
            left_index.append(LEFT_EYE_POINT_2[24 - i])

    for i in range(len(LEFT_EYE_POINT_3)):
        if i <= 8:
            left_index.append(LEFT_EYE_POINT_3[8 - i])
        else:
            left_index.append(LEFT_EYE_POINT_3[24 - i])

    # 입
    for i in range(int(len(MOUSE_POINT_1) / 2)):
        if i == 5:
            middle_index.append(MOUSE_POINT_1[i])
            middle_index.append(MOUSE_POINT_1[i + int(len(MOUSE_POINT_1) / 2)])
        elif i < 5:
            right_index.append(MOUSE_POINT_1[i])
            left_index.append(MOUSE_POINT_1[10 - i])
        else:
            right_index.append(MOUSE_POINT_1[25 - i])
            left_index.append(MOUSE_POINT_1[5 + i])

    for i in range(int(len(MOUSE_POINT_2) / 2)):
        if i == 5:
            middle_index.append(MOUSE_POINT_2[i])
            middle_index.append(MOUSE_POINT_2[i + int(len(MOUSE_POINT_2) / 2)])
        elif i < 5:
            right_index.append(MOUSE_POINT_2[i])
            left_index.append(MOUSE_POINT_2[10 - i])
        else:
            right_index.append(MOUSE_POINT_2[25 - i])
            left_index.append(MOUSE_POINT_2[5 + i])

    for i in range(int(len(MOUSE_POINT_3) / 2)):
        if i == 5:
            middle_index.append(MOUSE_POINT_3[i])
            middle_index.append(MOUSE_POINT_3[i + int(len(MOUSE_POINT_3) / 2)])
        elif i < 5:
            right_index.append(MOUSE_POINT_3[i])
            left_index.append(MOUSE_POINT_3[10 - i])
        else:
            right_index.append(MOUSE_POINT_3[25 - i])
            left_index.append(MOUSE_POINT_3[5 + i])

    for i in range(int(len(MOUSE_POINT_4) / 2)):
        if i == 5:
            middle_index.append(MOUSE_POINT_4[i])
            middle_index.append(MOUSE_POINT_4[i + int(len(MOUSE_POINT_4) / 2)])
        elif i < 5:
            right_index.append(MOUSE_POINT_4[i])
            left_index.append(MOUSE_POINT_4[10 - i])
        else:
            right_index.append(MOUSE_POINT_4[25 - i])
            left_index.append(MOUSE_POINT_4[5 + i])

    for i in range(int(len(MOUSE_POINT_5) / 2)):
        if i == 5:
            middle_index.append(MOUSE_POINT_5[i])
            middle_index.append(MOUSE_POINT_5[i + int(len(MOUSE_POINT_5) / 2)])
        elif i < 5:
            right_index.append(MOUSE_POINT_5[i])
            left_index.append(MOUSE_POINT_5[10 - i])
        else:
            right_index.append(MOUSE_POINT_5[25 - i])
            left_index.append(MOUSE_POINT_5[5 + i])

    for i in range(int(len(NOSE_POINT_1) / 2)):
        if i == 0:
            middle_index.append(NOSE_POINT_1[i])
            middle_index.append(NOSE_POINT_1[i + int(len(NOSE_POINT_1) / 2)])
        else:
            right_index.append(NOSE_POINT_1[i])
            left_index.append(NOSE_POINT_1[len(NOSE_POINT_1) - i])

    for i in MIDDLE_NOSE_LINE:
        middle_index.append(i)

    for i in RIGHT_NOSE_POINT:
        right_index.append(i)

    for i in LEFT_NOSE_POINT:
        left_index.append(i)

    for i in RIGHT_IRIS:
        right_index.append(i)

    for i in range(len(LEFT_IRIS)):
        if i % 2 == 1:
            left_index.append(LEFT_IRIS[4 - i])
        else:
            left_index.append(LEFT_IRIS[i])
relocation_list()

# For static images:
IMAGE_FILES = glob.glob('../img/longhair.jpg')
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()

    for face_landmarks in results.multi_face_landmarks:
      #print('face_landmarks:', face_landmarks)

      lendmarks = OUTLINE_POINTS_4

      # print(len(lendmarks))
      # for idx in lendmarks:
      # for idx in middle_index:
      #   loc_x = int(face_landmarks.landmark[idx].x * image.shape[1])
      #   loc_y = int(face_landmarks.landmark[idx].y * image.shape[0])
      #   cv2.circle(annotated_image,(loc_x, loc_y), 1, (255,255,0), 2)
      # for idx in range(len(left_index)):
      #   loc_x = int(face_landmarks.landmark[left_index[idx]].x * image.shape[1])
      #   loc_y = int(face_landmarks.landmark[left_index[idx]].y * image.shape[0])
      #   cv2.circle(annotated_image,(loc_x, loc_y), 1, (255,0,255), 2)
      #
      #   loc_x = int(face_landmarks.landmark[right_index[idx]].x * image.shape[1])
      #   loc_y = int(face_landmarks.landmark[right_index[idx]].y * image.shape[0])
      #   cv2.circle(annotated_image,(loc_x, loc_y), 1, (255,0,0), 2)
      #
      #   print(right_index[idx],left_index[idx])



      for idx in range(468,478):
          loc_x = int(face_landmarks.landmark[idx].x * image.shape[1])
          loc_y = int(face_landmarks.landmark[idx].y * image.shape[0])
          cv2.circle(annotated_image,(loc_x, loc_y), 1, (255,0,0), 2)
          cv2.imshow('img', annotated_image)
          cv2.waitKey()

    cv2.imshow('img',annotated_image)
    cv2.waitKey()

sum_list = OUTLINE_POINTS_1 + OUTLINE_POINTS_2 + OUTLINE_POINTS_3+OUTLINE_POINTS_4+RIGHT_OUTLINE_POINT+LEFT_OUTLINE_POINT+\
           RIGHT_EYE_POINT_1+RIGHT_EYE_POINT_2+RIGHT_EYE_POINT_3+LEFT_EYE_POINT_1+LEFT_EYE_POINT_2+LEFT_EYE_POINT_3+MOUSE_POINT_1+\
           MOUSE_POINT_2+MOUSE_POINT_3+MOUSE_POINT_4+MOUSE_POINT_5+NOSE_POINT_1+MIDDLE_NOSE_LINE+RIGHT_NOSE_POINT+LEFT_NOSE_POINT

a = sum_list
x = [] # 처음 등장한 값인지 판별하는 리스트
new_a = [] # 중복된 원소만 넣는 리스트

for i in a:
    if i not in x: # 처음 등장한 원소
        x.append(i)
    else:
        if i not in new_a: # 이미 중복 원소로 판정된 경우는 제외
            new_a.append(i)
