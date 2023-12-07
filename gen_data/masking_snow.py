import cv2
import os
import random

SNOW = "data/"
MASK = "snow_mask/"
OUTPUT = "snowed/"

snow_list = os.listdir(SNOW)
mask_list = os.listdir(MASK)
cnt = 0
for i in range(4000,8000):
    image = cv2.imread(SNOW + snow_list[i])
    number = random.randint(1, len(mask_list) - 1)
    mask = cv2.imread(MASK + mask_list[number], cv2.IMREAD_GRAYSCALE)

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.add(image, mask3)

    cv2.imwrite(OUTPUT + snow_list[i], masked_image)

