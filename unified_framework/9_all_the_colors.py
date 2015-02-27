#! /usr/bin/env python2.7
# -*- coding: iso-8859-1 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import time

Dir = '/home/omari/Desktop/Python/language/Simultanous_learning_and_ground/images/'

#============================== Constants ======================================#


tests = 1			# for monte carlo test
n_examples = 2000			# number of examples in each test
plotting = 1			# choose to plot graphs or not
time = 1			# time to view each example
object_number_max = 4		# maximum number of objects allowed in one scene

display_frame = 5		# number of examples to update the robot brain
rotation_inc = 50		# rotation increament for the robot brain view



#-------------------------------------------------------------------------------------#
def HSV_to_RGB(H,S,V):

	if S>1:
		S=np.zeros(1)+1
	if S<0:
		S=-S
	if V>1:
		V=np.zeros(1)+1
	if V<0:
		V=-V
	if H<0:
		H = H+360
	if H>360:
		H = H-360
	#http://www.rapidtables.com/convert/color/hsv-to-rgb.htm
	C = V * S
	X = C * (1 - np.abs(np.mod((H / 60.0),2.0) - 1))
	m = V - C
	if H>=0 and H<60:
		R=C
		G=X
		B=0
	elif H>=60 and H<120:
		R=X
		G=C
		B=0
	elif H>=120 and H<180:
		R=0
		G=C
		B=X
	elif H>=180 and H<240:
		R=0
		G=X
		B=C
	elif H>=240 and H<300:
		R=X
		G=0
		B=C
	elif H>=300 and H<=360:
		R=C
		G=0
		B=X
	return int((R+m)*255),int((G+m)*255),int((B+m)*255)


#-----------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
	
	name = ['red','green','blue','yellow','aqua','black','maroon','white','silver','fuchsia','purple','olive']
	img = np.zeros(shape=(540, 50+720, 3), dtype=np.uint8)+255 
	object_radius = 20
	#o_loc,o_color,img = scene_gnerator(object_number_max)
	object_locations = []
	object_colors = []
	counter = 0
	C = [[0,1,1],[120,1,1],[240,1,1],[60,1,1],[180,1,1],[0,0,0],[0,1,.5],[0,0,1],[0,0,.75],[300,1,1],[300,1,.5],[60,1,.5]]
	for obj in range(len(C)):
		for i in range(5):
			for j in range(5):
				for k in range(5):
					obj_2 = np.abs([obj*2*object_radius+2*object_radius , 50+counter*5+2*object_radius])
					counter += 1
					object_locations.append([obj_2])	# first object
					H = C[obj][0]+5*(j-2.5)
					S = C[obj][1]+.05*(k-5)
					V = C[obj][2]+.05*(i-5)
					print H,S,V
					print HSV_to_RGB(H,S,V)
					object_colors.append(HSV_to_RGB(H,S,V))

		counter = 0

	for j in range(len(object_locations)):
		c = object_colors[j]
		l = object_locations[j][0]
		#cv2.circle(img,(int(l[1])+object_radius,int(l[0])+object_radius),object_radius+2,(0,0,0),-1)
		cv2.circle(img,(int(l[1])+object_radius,int(l[0])+object_radius),object_radius,(c[2],c[1],c[0]),-1)

	counter = 0
	for j in name:
		cv2.putText(img,j,(10,counter*2*object_radius+60), cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,0),1)
		counter += 1

	cv2.imwrite(Dir+'All colours.png',img)
	cv2.imshow('Table',img)
	k = cv2.waitKey(10000) & 0xFF









