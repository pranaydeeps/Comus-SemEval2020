import cv2
import numpy as np
import os
import tqdm

template_list = os.listdir('data/TEMPLATES')
images_list = os.listdir('data/Meme_images')


test_image = images_list[40]
original = cv2.imread("data/Meme_images/" + test_image)
max_good_points = 0
correct_template = ''

print('Iterating Through Templates for Image: {}'.format(test_image))
for template in tqdm.tqdm(template_list):
    try:
        image_to_compare = cv2.imread("data/TEMPLATES/" + template)


        sift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(original, None)
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        ratio = 0.6
        for m, n in matches:
            if m.distance < ratio*n.distance:
                good_points.append(m)

        if len(good_points) > max_good_points:
            max_good_points = len(good_points)
            correct_template = template
    except:
        print('Excepted, moving on.')

# result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

# cv2.imshow("result", result)
identified_template = cv2.imread("data/TEMPLATES/" + correct_template)
cv2.imshow("Original", original)
cv2.imshow("Identified Template", identified_template)
cv2.waitKey(0)
cv2.destroyAllWindows()