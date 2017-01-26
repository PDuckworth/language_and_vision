import cv2
import numpy as np
import colorsys

f = open("/home/omari/Baxter_Dataset_Final/all_colours/all_colours.txt", 'r')
f2 = open("/home/omari/Baxter_Dataset_Final/all_colours/all_colours_hsv.txt", 'w')
for i in f:
    r,g,b = i.split('\n')[0].split(',')
    # print r,g,b
    h,s,v = colorsys.rgb_to_hsv(int(r)/255.0, int(g)/255.0, int(b)/255.0)
    f2.write('%.4f,%.4f,%.4f\n' %(h,s,v))
    # print colorsys.hsv_to_rgb(h,s,v)[0]*255
    # break
f2.close()
