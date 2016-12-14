from os import listdir
from os.path import isfile, join
import cv2
import numpy as np


dir1_sk = '/home/omari/Downloads/Demo_images/11:16:33_9021321f-bb3d-5540-8066-9dc68c8cb7f2/rgb_sk/'
dir1_cp = '/home/omari/Downloads/Demo_images/11:16:33_9021321f-bb3d-5540-8066-9dc68c8cb7f2/rgb_sk_cpm/'
images1 = sorted([f for f in listdir(dir1_sk) if isfile(join(dir1_sk, f))])

dir2_sk = '/home/omari/Downloads/Demo_images/12:07:21_54af6394-1a5e-57ca-9d88-1c7e360dc19b/rgb_sk/'
dir2_cp = '/home/omari/Downloads/Demo_images/12:07:21_54af6394-1a5e-57ca-9d88-1c7e360dc19b/rgb_sk_cpm/'
images2 = sorted([f for f in listdir(dir2_sk) if isfile(join(dir2_sk, f))])

dir3_sk = '/home/omari/Downloads/Demo_images/12:16:28_4a634c9a-ff81-5558-b004-e4a890178c92/rgb_sk/'
dir3_cp = '/home/omari/Downloads/Demo_images/12:16:28_4a634c9a-ff81-5558-b004-e4a890178c92/rgb_sk_cpm/'
images3 = sorted([f for f in listdir(dir3_sk) if isfile(join(dir3_sk, f))])

dir = '/home/omari/Downloads/Demo_images/images/'

names = cv2.imread(dir+'names.png')
counter = 1
def get_frame(f):
    if f < 10:
        return '000'+str(f)
    elif f < 100:
        return '00'+str(f)
    elif f < 1000:
        return '0'+str(f)
    elif f < 10000:
        return str(f)

for img in images1:
    im1 = cv2.imread(dir1_sk+img)
    im2 = cv2.imread(dir1_cp+img)
    sh = im1.shape
    im = np.zeros((sh[0]*2+20,sh[1]+150,3),dtype=np.uint8)+255
    im[:,0:150,:] = names
    im[:sh[0],150:150+sh[1],:]=im1
    im[sh[0]+20:20+2*sh[0],150:150+sh[1],:]=im2
    cv2.imshow('final',im)
    frame = get_frame(counter)
    counter += 1
    cv2.imwrite(dir+'img_'+frame+'.jpg',im)
    cv2.waitKey(20)

for img in images3:
    im1 = cv2.imread(dir3_sk+img)
    im2 = cv2.imread(dir3_cp+img)
    sh = im1.shape
    im = np.zeros((sh[0]*2+20,sh[1]+150,3),dtype=np.uint8)+255
    im[:,0:150,:] = names
    im[:sh[0],150:150+sh[1],:]=im1
    im[sh[0]+20:20+2*sh[0],150:150+sh[1],:]=im2
    cv2.imshow('final',im)
    frame = get_frame(counter)
    counter += 1
    cv2.imwrite(dir+'img_'+frame+'.jpg',im)
    cv2.waitKey(20)
    
for img in images2:
    im1 = cv2.imread(dir2_sk+img)
    im2 = cv2.imread(dir2_cp+img)
    sh = im1.shape
    im = np.zeros((sh[0]*2+20,sh[1]+150,3),dtype=np.uint8)+255
    im[:,0:150,:] = names
    im[:sh[0],150:150+sh[1],:]=im1
    im[sh[0]+20:20+2*sh[0],150:150+sh[1],:]=im2
    cv2.imshow('final',im)
    frame = get_frame(counter)
    counter += 1
    cv2.imwrite(dir+'img_'+frame+'.jpg',im)
    cv2.waitKey(20)
