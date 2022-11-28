import cv2
from skimage.metrics import structural_similarity as ssim

other_image_name = ['./test_image/front01+01.png', './test_image/front02+02.png',
                    './test_image/front02+03.png', './test_image/front03+03.png']
base_image = './test_image/base2.png'

for other in other_image_name:
    imageA = cv2.imread(base_image)
    imageA = cv2.resize(imageA, (800,800))

    imageB = cv2.imread(other)
    imageB = cv2.resize(imageB, (800,800))

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = ssim(grayA, grayB, full=True)

    print(f"Similarity: {score:.5f}")

    cv2.imshow('base', grayA)
    cv2.imshow('other', grayB)

    cv2.waitKey()
    cv2.destroyAllWindows()