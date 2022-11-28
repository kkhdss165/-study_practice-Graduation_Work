import cv2
import os
import numpy as np

def get_mask(file_name):
    img = cv2.imread(file_name)
    target = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(target,1,255, cv2.THRESH_BINARY)

    contour2, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x = []
    y = []
    for i in contour2:
        if len(i) > 5 :
            for j in i:
                x.append(j[0][0])
                y.append(j[0][1])

    img = cv2.rectangle(img, (min(x), min(y)), (max(x), max(y)), (255,0,0), 1)
    target_mask = thresh1[min(y):max(y)+1, min(x):max(x)+1]

    return target_mask

dir_path = './ssim'
target_path = './notitle.png'

target_mask = get_mask(target_path)
total = []
for (root, directories, files) in os.walk(dir_path):
    for file in files:
        if '.png' in file:
            file_path = os.path.join(root, file)
            print(file_path)
            loop_mask = get_mask(file_path)

            cv2.imshow('target_mask',target_mask)
            cv2.imshow('loop_mask',loop_mask)

            t_h, t_w = target_mask.shape
            l_h, l_w = loop_mask.shape

            re_h = t_h + l_h
            re_w = t_w + l_w


            new_canvas_loop = np.zeros((re_h, re_w), np.uint8)

            new_canvas_loop[re_h//2-l_h//2 : re_h//2-l_h//2 + l_h, re_w//2-l_w//2 : re_w//2-l_w//2 + l_w] = loop_mask

            maxscore = None
            min_h = None
            min_w = None
            value1 = None
            value2 = None

            for height in range(0,re_h-t_h,20):
                for width in range(0,re_w-t_w,20):
                    new_canvas_target = np.zeros((re_h, re_w), np.uint8)
                    new_canvas_target[height:height+t_h, width:width+t_w] = target_mask

                    bit_xor = cv2.bitwise_xor(new_canvas_loop, new_canvas_target)
                    bit_and = cv2.bitwise_and(new_canvas_loop, new_canvas_target)

                    and_sum = np.sum(bit_and)
                    xor_sum = np.sum(bit_xor)

                    score = and_sum - xor_sum
                    print(and_sum,xor_sum, score)
                    # print(sum(sum(bit_and)),sum(sum(bit_xor)), sum(sum(bit_and)) - sum(sum(bit_xor)))
                    # cv2.imshow("new_canvas_loop", new_canvas_loop)
                    # cv2.imshow("new_canvas_target", new_canvas_target)
                    #
                    # cv2.imshow("bit_xor", bit_xor)
                    # cv2.imshow("bit_and", bit_and)
                    # cv2.waitKey()

                    if maxscore == None or maxscore < score:
                        maxscore = score
                        min_h = height
                        min_w = width
                        value1 = and_sum
                        value2 = xor_sum

            new_canvas_target = np.zeros((re_h, re_w), np.uint8)
            new_canvas_target[min_h:min_h + t_h, min_w:min_w + t_w] = target_mask

            bit_xor = cv2.bitwise_xor(new_canvas_loop, new_canvas_target)
            bit_and = cv2.bitwise_and(new_canvas_loop, new_canvas_target)
            cv2.imshow("new_canvas_loop", new_canvas_loop)
            cv2.imshow("new_canvas_target", new_canvas_target)

            cv2.imshow("bit_xor", bit_xor)
            cv2.imshow("bit_and", bit_and)
            print(file_path, "and: ",value1, "xor: ",value2, score)
            cv2.waitKey()

            cv2.destroyAllWindows()

            total.append(maxscore)

print(total.index(max(total)))