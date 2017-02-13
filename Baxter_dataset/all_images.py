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
    dir_dst = "/home/omari/Datasets/AAAI-17-Baxter-dataset/scene"+folder

    # remove HSV clusters
    # files = sorted(glob.glob(dir_dst+"/objects/"+"HSV_cluster_*"))
    # for file in files:
    #     os.remove(file)

    # change .png to ,jpg
    for d in ["/rgb_cam"]:#,"/rgb_kinect","/objects","/rgb_lefthand","/rgb_righthand","/robot_state","/table_pointcloud","/tabletop_pointcloud","/object_tracks"]:
        files = sorted(glob.glob(dir_dst+d+"/*.jpg"))
        count = 0
        for file in files:
            count+=1
            # print count
            if np.mod(count,5.0)==0:
                # print count
                img = cv2.imread(file)
                cv2.imshow("img",img)
                cv2.waitKey(10)
                img = cv2.resize(img,None,fx=.6, fy=.6, interpolation = cv2.INTER_CUBIC)
                cv2.imwrite("/home/omari/Datasets/AAAI-17-Baxter-dataset/all_images/img"+str(counter)+".jpg",img)
                counter+=1
    #             cv2.imwrite(file.split(".")[0]+".jpg",img)
    #             os.remove(file)
    #         except Exception as e:
    #             pass
    #
    # shutil.make_archive(dir_dst, 'zip', dir_dst)
