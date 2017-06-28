import cv2
import numpy as np

shapes = '/home/omari/Datasets/Baxter_Dataset_final/features/shapes/cluster_images/'
#colors = '/home/omari/Datasets/ECAI_dataset/colours/'
#actions = '/home/omari/Datasets/ECAI_dataset/actions/'
#objects = '/home/omari/Datasets/ECAI_dataset/objects/'
image_dir = "/home/omari/Dropbox/Thesis/writing/Chapter7/Chapter7Figs/PNG/LUCAS-visual-concepts.jpg"

shape_im = ['7_cluster', '26_cluster']
#color_im = ['5_cluster', '4_cluster', '9_cluster']
#action_im = ['3_cluster', '6_cluster', '11_cluster']
#object_im = ['0_cluster', '1_cluster', '2_cluster']
all_images = [shape_im]#, color_im, action_im, object_im]
all_dir = [shapes]#, colors, actions, objects]
text = ["shape"]#, "colour", "action", "object"]

im_len = 60
th = 20
font = cv2.FONT_HERSHEY_SIMPLEX

image = np.zeros((im_len*5*2+th*2,im_len*5*4+th*3,3),dtype=np.uint8)+255

for c1,i in enumerate(all_images):
    dir = all_dir[c1]
    t = text[c1]
    for c2,j in enumerate(i):
        im = cv2.imread(dir+j+".jpg")
        image[c2*(im_len*5)+c2*th:(c2+1)*(im_len*5)+c2*th, c1*(im_len*5)+c1*th:(c1+1)*(im_len*5)+c1*th, :] = im
        cv2.putText(image,t+'_'+str(c2),   (100+c1*im_len*5, (c2+1)*(im_len*5)+(c2)*th+th*5/6),    font, .8,(0,0,0),2)
        print t
cv2.imwrite(image_dir,image)
cv2.imshow("test",image)
cv2.waitKey(1000)
