import numpy as np
import cv2

dirr = '/home/omari/Python/Python_images/language_and_vision/images2/'

for i in np.linspace(4,249,50):
        f = int(i)
        if f<10: frame = '00'+str(f)
        elif f<100: frame = '0'+str(f)
        elif f<1000: frame = str(f)
        print frame
        img = np.zeros((430,420*2+420,3),dtype=np.uint8)+255
        img1 = cv2.imread(dirr+'HSV-'+frame+'.png')
        img2 = cv2.imread(dirr+'Distance-'+frame+'.png')
        img3 = cv2.imread(dirr+'Direction-'+frame+'.png')
        img[50:380,0:420,:] = img1[50:380,70:490,:]
        img[50:380,420:420+420,:] = img2[50:380,70:490,:]
        img[50:380,420+420:420*2+420,:] = img3[50:380,70:490,:]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'HSV',(190,30), font, 1,(0,0,0),1)
        cv2.putText(img,'Distance',(580,30), font, 1,(0,0,0),1)
        cv2.putText(img,'Direction',(1000,30), font, 1,(0,0,0),1)
        cv2.putText(img,'Example number : '+str(f+1),(40,410), font, 1,(0,0,0),1)
        
        cv2.imwrite(dirr+'test-'+frame+'.png',img)
