import numpy as np
import cv2

folder = '93'
maxi = 73
cam = '/home/omari/Datasets/scene'+folder+'/cam/cam_0'
kin = '/home/omari/Datasets/scene'+folder+'/kinect_rgb/Kinect_0'
trk = '/home/omari/Datasets/scene'+folder+'/tracking/scene_0'
rhc = '/home/omari/Datasets/scene'+folder+'/RH_rgb/Right_0'
# rhc = '/home/omari/Datasets/scene'+folder+'/LH_rgb/Left_0'

logo1 = cv2.imread('/home/omari/Datasets/all_colours/cam.png')
logo2 = cv2.imread('/home/omari/Datasets/all_colours/rhc.png')
# logo2 = cv2.imread('/home/omari/Datasets/all_colours/lhc.png')
logo3 = cv2.imread('/home/omari/Datasets/all_colours/kin.png')
logo4 = cv2.imread('/home/omari/Datasets/all_colours/trk.png')

txt = 70
sp = 20
for i in range(1,maxi+1):
    img_tot = np.zeros((txt*2+480*2+sp,640*2+sp,3),dtype=np.uint8)+255
    if i < 10:
        file = '00'+str(i)
    elif i < 100:
        file = '0'+str(i)
    elif i < 1000:
        file = str(i)

    img_cam = cv2.imread(cam+file+'.png')
    img_kin = cv2.imread(kin+file+'.png')[:,80:-80,:]
    img_kin = cv2.resize(img_kin,None,fx=640.0/img_kin.shape[1], fy=480.0/img_kin.shape[0], interpolation = cv2.INTER_CUBIC)
    img_trk = cv2.imread(trk+file+'.png')[50:-50,130:-300,:]
    img_trk = cv2.resize(img_trk,None,fx=640.0/img_trk.shape[1], fy=480.0/img_trk.shape[0], interpolation = cv2.INTER_CUBIC)
    img_rhc = cv2.imread(rhc+file+'.png')[:,50:590,:]
    img_rhc = cv2.resize(img_rhc,None,fx=640.0/540, fy=480.0/400, interpolation = cv2.INTER_CUBIC)
    # print img_cam.shape
    img_tot[0+txt:480+txt,0:640,:] = img_cam
    img_tot[0+txt:480+txt,640+sp:,:] = img_rhc
    img_tot[480+sp+txt*2:,0:640:,:] = img_kin
    img_tot[480+sp+txt*2:,640+sp:,:] = img_trk
    font = cv2.FONT_HERSHEY_SIMPLEX

    img_tot[0:0+txt,0:640,:] = logo1
    img_tot[0:0+txt,640+sp:640*2+sp,:] = logo2
    img_tot[480+sp+txt:480+sp+2*txt,0:640,:] = logo3
    img_tot[480+sp+txt:480+sp+2*txt,640+sp:640*2+sp,:] = logo4
    # cv2.putText(img_tot,'OpenCV',(0,500), font, 4,(0,0,0),2)
    cv2.imshow('img',img_tot)
    cv2.imwrite('/home/omari/Datasets/videos/video'+folder+'_'+str(i)+'.png',img_tot)
    cv2.waitKey(20)




# print img.shape
# sp = 20
# total_img = np.zeros((806,480*4+sp*3,3),dtype=np.uint8)+255
# c = 0
# for i in [1,2,4,5]:
#     img = cv2.imread(dir1+'IMG_'+str(i)+'.jpg')
#     img = cv2.resize(img,None,fx=.2, fy=.2, interpolation = cv2.INTER_CUBIC)
#     img = img[:,75:555,:]
#     total_img[:,480*c+sp*c:480*(c+1)+sp*c,:] = img
#     c+=1
#
#
# total_img = total_img[50:,:,:]
# cv2.imshow('img',total_img)
# cv2.imwrite(dir1+'example4.png',total_img)
# cv2.waitKey(2000)
