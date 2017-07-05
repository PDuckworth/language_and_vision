import cv2
import numpy as np

shapes = '/home/omari/Datasets/jivko_dataset/features/shapes/cluster_images/'
colors = '/home/omari/Datasets/jivko_dataset/features/colours/'
# locations = '/home/omari/Datasets/jivko_dataset/features/locations/cluster_images/'
# distances = '/home/omari/Datasets/Baxter_Dataset_final/features/distances/cluster_images/'
#colors = '/home/omari/Datasets/ECAI_dataset/colours/'
#actions = '/home/omari/Datasets/ECAI_dataset/actions/'
#objects = '/home/omari/Datasets/ECAI_dataset/objects/'
image_dir = "/home/omari/Dropbox/Thesis/writing/Chapter7/Chapter7Figs/PNG/sinapov-visual-concepts.jpg"

shape_im = ['9_cluster', '3_cluster']
color_im = ['2_cluster', '3_cluster']
# location_im = ['2_cluster', '5_cluster']
# distance_im = ['0_cluster', '2_cluster']
all_images = [shape_im, color_im] #, location_im, distance_im]
all_dir = [shapes, colors] #, locations, distances]#, actions, objects]
text = ["shape", "colour"] #, "location", "distance"]#, "action", "object"]

im_len = 60
th = 40
font = cv2.FONT_HERSHEY_SIMPLEX

image = np.zeros((im_len*5*2+th*2,im_len*5*3+th*2,3),dtype=np.uint8)+255

for c1,i in enumerate(all_images):
    dir_ = all_dir[c1]
    t = text[c1]
    for c2,j in enumerate(i):
        im = cv2.imread(dir_+j+".jpg")
        print dir_+j+".jpg"
        image[c2*(im_len*5)+c2*th:(c2+1)*(im_len*5)+c2*th, c1*(im_len*5)+c1*th:(c1+1)*(im_len*5)+c1*th, :] = im
        cv2.putText(image,t+'_'+str(c2),   (100+c1*(im_len*5)+c1*th, (c2+1)*(im_len*5)+(c2)*th+th*1/2),    font, .8,(0,0,0),2)
        # print t
cv2.imwrite(image_dir,image)
cv2.imshow("test",image)
cv2.waitKey(3000)
