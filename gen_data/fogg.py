import cv2
import os
import numpy as np
import random

def gen_haze(img, depth_img):
    
    depth_img_3c = np.zeros_like(img)
    depth_img_3c[:,:,0] = depth_img
    depth_img_3c[:,:,1] = depth_img
    depth_img_3c[:,:,2] = depth_img

    beta = 3
    norm_depth_img = depth_img_3c/255
    trans = np.exp(-norm_depth_img*beta)

    A = 255
    hazy = img*trans + A*(1-trans)
    hazy = np.array(hazy, dtype=np.uint8)
    
    return hazy

INPUT = "data/"
DEPTH = "depth/"
OUTPUT = ""
#
list_image = os.listdir(INPUT)
list_depth = os.listdir(DEPTH)

# Load the image
# image = cv2.imread(INPUT + list_image[0])
# depth = cv2.imread(DEPTH + list_depth[0], cv2.IMREAD_GRAYSCALE)

# image = gen_haze(image, depth)
# print (image)
# # Display image
# cv2.imshow('Test', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

for i in range(1):
    image = cv2.imread(INPUT + list_image[i])
    depth = cv2.imread(DEPTH+ list_depth[i], cv2.IMREAD_GRAYSCALE)

    image = gen_haze(image, depth)

    cv2.imwrite(OUTPUT + list_image[i], image)



#preferences
# @article{tran2022novel,
#   title={A novel encoder-decoder network with guided transmission map for single image dehazing},
#   author={Tran, Le-Anh and Moon, Seokyong and Park, Dong-Chul},
#   journal={Procedia Computer Science},
#   volume={204},
#   pages={682--689},
#   year={2022},
#   publisher={Elsevier}
# }