import numpy as np
import os
import shutil
import glob
import cv2

dir_dst = "/home/omari/Datasets/Baxter_Dataset_final/all_images/img"

count = 0
th = 200
for d in ["51","152"]:

    img = cv2.imread(dir_dst+d+".png")
    img2 = img[-th:,:,:]*0+255
    if count == 0:
        img_final = img
        img_final = np.concatenate((img_final, img2), axis=0)
    else:
        img_final = np.concatenate((img_final, img), axis=0)
        img_final = np.concatenate((img_final, img2), axis=0)
    count+=1

# img_final = img_final[:,:-th,:]
cv2.imwrite("/home/omari/Datasets/Baxter_Dataset_final/all_images/LUCAS-dataset.png",img_final)
