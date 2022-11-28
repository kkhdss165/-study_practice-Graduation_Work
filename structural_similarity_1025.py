from skimage.metrics import structural_similarity
import cv2

other_image_name = ['./test_image/front01+01.png', './test_image/front02+02.png',
                    './test_image/front02+03.png', './test_image/front03+03.png']
base_image = './test_image/base2.png'

def orb_sim(img1, img2):
    # SIFT is no longer available in cv2 so using ORB
    orb = cv2.ORB_create()

    # detect keypoints and descriptors
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # define the bruteforce matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # perform matches.
    matches = bf.match(desc_a, desc_b)
    # Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)

def structural_sim(img1, img2):

  sim, diff = structural_similarity(img1, img2, full=True)
  return sim



for other in other_image_name:
    imageA = cv2.imread(base_image)
    imageA = cv2.resize(imageA, (800,800))

    imageB = cv2.imread(other)
    imageB = cv2.resize(imageB, (800,800))

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    orb_similarity = orb_sim(grayA, grayB)
    score = structural_sim(grayA, grayB)

    print("Similarity using ORB is: ", orb_similarity)
    print(f"Similarity: {score:.5f}")

    print(orb_similarity * score)

    cv2.imshow('base', grayA)
    cv2.imshow('other', grayB)

    cv2.waitKey()
    cv2.destroyAllWindows()