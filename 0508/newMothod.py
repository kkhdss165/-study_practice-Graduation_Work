import numpy as np
import cv2

color_dic = {0:[255,255,255],
             1:[255,0,0],
             2:[0,0,255],
             3:[0,255,0],
             4:[0,255,255],
             5:[255,0,255],
             6:[255,255,0],
             7:[0,0,0]}

K = 8 # 군집화 갯수(16컬러) ---①
img = cv2.imread('../img/hyunbin.png')
img_y, img_x, _ = img.shape
data = img.reshape((-1,3)).astype(np.float32)
# 반복 중지 요건 ---③
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 평균 클러스터링 적용 ---④
ret,label,center=cv2.kmeans(data,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# 중심 값을 정수형으로 변환 ---⑤
center = np.uint8(center)
print(center)
# 각 레이블에 해당하는 중심값으로 픽셀 값 선택 ---⑥
res = center[label.flatten()]
print(label.flatten())
# 원본 영상의 형태로 변환 ---⑦
res = res.reshape((img.shape))
# 결과 출력 ---⑧
cv2.imshow('KMeans Color',res)
cv2.waitKey(0)

temp = np.zeros_like(img)

print(center[0])

list_center = center.tolist()
print(list_center)

list_res = res.tolist()


cv2.imshow('temp',temp)
cv2.waitKey(0)
for i in range(img_y):
    for j in range(img_x):
        value = list_res[i][j]
        index = list_center.index(value)
        temp[i][j] = color_dic[index]

cv2.imshow('temp', temp)
cv2.waitKey(0)
cv2.destroyAllWindows()