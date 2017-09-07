import numpy as np
import os
import shutil
import glob
import cv2
# import zipfile

counter = 0
for i in range(1,33):
    folder = str(i)
    # dir_src = "/home/omari/Datasets/Baxter_Dataset_final/scene"+folder
    dir_dst = "/home/omari/Datasets/jivko_dataset/t1/obj_"+folder+"/trial_1/"

    for d in ["drop","grasp","lift","push"]:
        files = glob.glob(dir_dst+d+"/vision_data/*.jpg")

        # print dir_dst+d+"/vision_data/"
        count = 0
        count2 = 0
        for i in range(len(files)):
            file = glob.glob(dir_dst+d+"/vision_data/test"+str(i)+"_*.jpg")
            img = cv2.imread(file[0])
            # img = img[:,:-140,:]
            img[:,-10:,:] = 255
            if np.mod(count,len(files)/3)==0:
                count2+=1
                if count2 > 3:
                    continue
                cv2.imshow("img",img)
                cv2.waitKey(10)
                if count == 0:
                    img_final = img
                else:
                    img_final = np.concatenate((img_final, img), axis=1)
            count+=1
        img_final = np.concatenate((img_final, img), axis=1)
        img_final = img_final[:,:-10,:]
        cv2.imwrite("/home/omari/Datasets/jivko_dataset/all_images/img"+str(counter)+".png",img_final)
        counter+=1
