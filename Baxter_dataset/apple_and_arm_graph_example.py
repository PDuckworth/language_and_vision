import numpy as np
import cv2

apple = "/home/omari/Dropbox/presentations/AAAI-17/pics/apple2.jpg"
arm = "/home/omari/Dropbox/presentations/AAAI-17/pics/arm.png"
g0 = "/home/omari/Dropbox/presentations/AAAI-17/pics/Simple_graph0.png"
g1 = "/home/omari/Dropbox/presentations/AAAI-17/pics/Simple_graph1.png"
g2 = "/home/omari/Dropbox/presentations/AAAI-17/pics/Simple_graph2.png"
img_a = cv2.imread(apple)
img_a = img_a[630:800,78:212,:]
img_r = cv2.imread(arm)
img_g = cv2.imread(g1)
# img_r[img_r>254] = 254
# img_a+=img_r

# print img_r.shape

count = 1

# for a in range(730,800,2):
#     i=400
#     img_r2 = img_g.copy()
#     # img_r2 = np.zeros((img_r.shape),dtype=np.uint8)+255
#     img_r2[:i+4,-310:,:] = img_r[800-i:,:,:]
#     img_r2[630:800,78-310:212-310,:] = img_a
#     cv2.imshow('img',img_r2)
#     cv2.waitKey(20)
#     cv2.imwrite("/home/omari/Dropbox/presentations/AAAI-17/pics/apple/"+str(count)+".jpg",img_r2)
#     count+=1

for i in range(400,800,5):
    img_r2 = np.zeros((img_r.shape),dtype=np.uint8)+255
    img_r2[:i+4,:,:] = img_r[800-i:,:,:]
    img_r2[630:800,78:212,:] = img_a

    img_r3 = img_g.copy()
    img_r3[:,-310:,:] = img_r2
    cv2.imshow('img',img_r3)
    cv2.waitKey(20)
    cv2.imwrite("/home/omari/Dropbox/presentations/AAAI-17/pics/apple/"+str(count)+".jpg",img_r3)
    count+=1


for i in range(120):
    cv2.waitKey(20)
    cv2.imwrite("/home/omari/Dropbox/presentations/AAAI-17/pics/apple/"+str(count)+".jpg",img_r3)
    count+=1


img_g = cv2.imread(g2)
for i in range(400,600,3):
    # img_r2 = np.zeros((img_r.shape),dtype=np.uint8)+255
    img_r2[:-5,:,:] = img_r2[5:,:,:]
    # img_r2[630:800,78:212,:] = img_a

    img_r3 = img_g.copy()
    img_r3[:,-310:,:] = img_r2
    cv2.imshow('img',img_r3)
    cv2.waitKey(20)
    cv2.imwrite("/home/omari/Dropbox/presentations/AAAI-17/pics/apple/"+str(count)+".jpg",img_r3)
    count+=1

for i in range(120):
    cv2.waitKey(20)
    cv2.imwrite("/home/omari/Dropbox/presentations/AAAI-17/pics/apple/"+str(count)+".jpg",img_r3)
    count+=1


# cv2.imshow("img1",img_a)
# cv2.imshow("img2",img_r)
