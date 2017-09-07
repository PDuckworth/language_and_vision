import numpy as np
import os
import shutil
import glob
import cv2
# import zipfile

counter = 0
for i in range(1,205):
    folder = str(i)
    # dir_src = "/home/omari/Datasets/Baxter_Dataset_final/scene"+folder
    dir_dst = "/home/omari/Datasets/ECAI Data/dataset_segmented_15_12_16/vid"+folder

    for d in ["/images"]:
        files = sorted(glob.glob(dir_dst+d+"/*.jpg"))
        count = 0
        count2 = 0
        for file in files:
            img = cv2.imread(file)
            # img = img[:,:-140,:]
            img[:,-10:,:] = 255
            if np.mod(count,len(files)/3)==0:
                count2+=1
                if count2 > 3:
                    continue
                cv2.imshow("img",img)
                cv2.waitKey(10)
                #img = cv2.resize(img,None,fx=.6, fy=.6, interpolation = cv2.INTER_CUBIC)
                if count == 0:
                    img_final = img
                else:
                    img_final = np.concatenate((img_final, img), axis=1)
            count+=1
    img_final = np.concatenate((img_final, img), axis=1)

    img_final = img_final[:,:-10,:]
    cv2.imwrite("/home/omari/Datasets/ECAI Data/all_images/img"+str(counter)+".png",img_final)
    counter+=1
