import numpy as np
import cv2

dir = "/home/omari/Datasets/metarooms/"

img1 = cv2.imread(dir+"1.png")
img1 = img1[30:-10,10:-10,:]
imga = cv2.imread(dir+"1a.png")
imga = imga[:,130:-100,:]
imga = cv2.resize(imga, (621,470))
imgb = cv2.imread(dir+"1b.png")
imgb = imgb[:,130:-100,:]
imgb = cv2.resize(imgb, (621,470))

sp1 = np.zeros((470,4,3),dtype=np.uint8)+255

vis1 = np.concatenate((img1.astype(np.uint8), sp1), axis=1)
vis1 = np.concatenate((vis1.astype(np.uint8), imga), axis=1)
vis1 = np.concatenate((vis1.astype(np.uint8), sp1), axis=1)
vis1 = np.concatenate((vis1.astype(np.uint8), imgb), axis=1)


img1 = cv2.imread(dir+"2.png")
img1 = img1[30:-10,10:-10,:]
imga = cv2.imread(dir+"2a.png")
imga = imga[:,130:-100,:]
imga = cv2.resize(imga, (621,470))
imgb = cv2.imread(dir+"2b.png")
imgb = imgb[:,130:-100,:]
imgb = cv2.resize(imgb, (621,470))

sp1 = np.zeros((470,4,3),dtype=np.uint8)+255

vis2 = np.concatenate((img1.astype(np.uint8), sp1), axis=1)
vis2 = np.concatenate((vis2.astype(np.uint8), imga), axis=1)
vis2 = np.concatenate((vis2.astype(np.uint8), sp1), axis=1)
vis2 = np.concatenate((vis2.astype(np.uint8), imgb), axis=1)


sp2 = np.zeros((20,vis1.shape[1],3),dtype=np.uint8)+255

vis = np.concatenate((vis1.astype(np.uint8), sp2), axis=0)
vis = np.concatenate((vis.astype(np.uint8), vis2), axis=0)


cv2.imshow('img',vis)
cv2.waitKey(2000)
cv2.imwrite(dir+'paper_images/objects.png',vis)
