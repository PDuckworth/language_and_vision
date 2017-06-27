import cv2
import numpy as np

faces = '/home/omari/Datasets/ECAI_dataset/faces/'
colors = '/home/omari/Datasets/ECAI_dataset/colours/'
image_dir = "/home/omari/Dropbox/Thesis/writing/Chapter7/Chapter7Figs/PNG/LUCIE-visual-concepts.png"

face_im = ['5_cluster', '10_cluster', '2_cluster', '30_cluster']
color_im = ['5_cluster', '7_cluster', '4_cluster', '9_cluster']
all_images = [face_im, color_im, face_im]
all_dir = [faces, colors, faces]
text = ["face", "colour", "action"]

im_len = 60
th = 50
font = cv2.FONT_HERSHEY_SIMPLEX

image = np.zeros((im_len*5*3+th*3,im_len*5*4,3),dtype=np.uint8)+255

for c1,i in enumerate(all_images):
    dir = all_dir[c1]
    t = text[c1]
    for c2,j in enumerate(i):
        im = cv2.imread(dir+j+".jpg")
        image[c1*(im_len*5)+c1*th:(c1+1)*(im_len*5)+c1*th, c2*(im_len*5):(c2+1)*(im_len*5), :] = im
        cv2.putText(image,t+'_'+str(c2),   (100+c2*im_len*5, (c1+1)*(im_len*5)+(c1)*th+th/2),    font, 1,(0,0,0),2)
cv2.imwrite(image_dir,image)
