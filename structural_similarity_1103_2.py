import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp


def get_mask(file_name, reverse = False):
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

    if reverse:
        target_mask = cv2.flip(target_mask)

    return target_mask

dir_path = './ssim'
target_path = './notitle5.png'
# target_path = './test_image/base3.png'

target_mask = get_mask(target_path)
total = []
for (root, directories, files) in os.walk(dir_path):
    for file in files:
        if '.png' in file:
            file_path = os.path.join(root, file)
            loop_mask = get_mask(file_path)

            sum = 0
            for num in range(1):

                t_h, t_w = target_mask.shape
                l_h, l_w = loop_mask.shape


                # 너비 길이 맞추고 조합
                max_w = max(t_w, l_w)
                target_mask = cv2.resize(target_mask, dsize =(0,0), fx= (max_w/t_w)*(1.0 + 0.2*num), fy= (max_w/t_w)*(1.0 + 0.2*num))
                loop_mask = cv2.resize(loop_mask, dsize =(0,0), fx=(max_w/l_w), fy= (max_w/l_w))

                t_h, t_w = target_mask.shape
                l_h, l_w = loop_mask.shape

                re_h = t_h + l_h
                re_w = t_w + l_w

                new_canvas_loop = np.zeros((re_h, re_w), np.uint8)

                new_canvas_loop[re_h//2-l_h//2 : re_h//2-l_h//2 + l_h, re_w//2-l_w//2 : re_w//2-l_w//2 + l_w] = loop_mask

                maxscore = None
                min_h = None
                min_w = None

                for height in range(0,re_h-t_h,20):
                    for width in range(0,re_w-t_w,20):
                        new_canvas_target = np.zeros((re_h, re_w), np.uint8)
                        new_canvas_target[height:height+t_h, width:width+t_w] = target_mask

                        bit_xor = cv2.bitwise_xor(new_canvas_loop, new_canvas_target)
                        bit_and = cv2.bitwise_and(new_canvas_loop, new_canvas_target)

                        and_sum = np.sum(bit_and)//255
                        xor_sum = np.sum(bit_xor)//255

                        score = xor_sum
                        # print(and_sum,xor_sum, score)
                        # print(sum(sum(bit_and)),sum(sum(bit_xor)), sum(sum(bit_and)) - sum(sum(bit_xor)))
                        # cv2.imshow("new_canvas_loop", new_canvas_loop)
                        # cv2.imshow("new_canvas_target", new_canvas_target)
                        #
                        # cv2.imshow("bit_xor", bit_xor)
                        # cv2.imshow("bit_and", bit_and)
                        # cv2.waitKey()

                        if maxscore == None or maxscore > score:
                            maxscore = score
                            min_h = height
                            min_w = width

                new_canvas_target = np.zeros((re_h, re_w), np.uint8)
                new_canvas_target[min_h:min_h + t_h, min_w:min_w + t_w] = target_mask

                bit_xor = cv2.bitwise_xor(new_canvas_loop, new_canvas_target)
                bit_and = cv2.bitwise_and(new_canvas_loop, new_canvas_target)
                bit_or = cv2.bitwise_or(new_canvas_loop, new_canvas_target)
                cv2.imshow("new_canvas_loop", new_canvas_loop)
                cv2.imshow("new_canvas_target", new_canvas_target)

                cv2.imshow("bit_xor", bit_xor)
                cv2.imshow("bit_and", bit_and)
                cv2.imshow("bit_or", bit_or)

                # print(file_path, "xor: ",score)

                # score = mse(bit_and, new_canvas_target)
                # score2 = mse(bit_or, new_canvas_target)
                score = mse(bit_or, bit_and)
                score2 = mse(new_canvas_loop, new_canvas_target)
                score3 = score + score2

                print(f"Similarity: {score:.5f}",f"Similarity: {score2:.5f}", f"sum: {score3:.5f}")
                # cv2.waitKey()
                sum += score3

            print(file_path, "sum: ", sum)
            total.append(sum)

print(total.index(min(total)))