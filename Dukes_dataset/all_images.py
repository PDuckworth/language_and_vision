import cv2

count = 0
for i in range(20,300):
    for j in ["0000","0004","0008","0012","0015"]:
        if i < 10:
            dir1 = "/home/omari/Datasets/robot_modified/scenes/"+str(i)+"/scene_000"+str(i)+"_frame_"+j+".png"
        elif i < 100:
            dir1 = "/home/omari/Datasets/robot_modified/scenes/"+str(i)+"/scene_00"+str(i)+"_frame_"+j+".png"
        elif i < 1000:
            dir1 = "/home/omari/Datasets/robot_modified/scenes/"+str(i)+"/scene_0"+str(i)+"_frame_"+j+".png"
        elif i < 10000:
            dir1 = "/home/omari/Datasets/robot_modified/scenes/"+str(i)+"/scene_"+str(i)+"_frame_"+j+".png"
        print dir1
        img = cv2.imread(dir1)
        img = img[:-200,100:-150,:]
        cv2.imwrite("/home/omari/Datasets/robot_modified/all_images/robot_"+str(count)+".png",img)
        count+=1
        # cv2.imshow("img",img)
        # cv2.waitKey(200)
