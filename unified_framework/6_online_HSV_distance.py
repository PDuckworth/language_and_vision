#! /usr/bin/env python2.7
# -*- coding: iso-8859-1 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import operator
import time
from Graph_functions import *
from itertools import product, combinations

Dir = '/home/omari/Desktop/Python/language/Simultanous_learning_and_ground/images/'
Dir2 = '/home/omari/Desktop/Python/language/Simultanous_learning_and_ground/images2/'

#============================== Constants ======================================#

object_radius = 20
map_size = 500
object_number_max = 2
n_examples = 2000
frames = 1
H_std = 5
color_number_max = 12
window = 1000
name_window = 150
th = 40
tests = 1
plotting = 1
time = 1
display_frame = 100
rotation_inc = 100
theta = np.linspace(-1 * np.pi, 1 * np.pi, 100)
zc1 = np.linspace(1, 1, 100)
xc1 = zc1 * np.sin(theta)
yc1 = zc1 * np.cos(theta)

zc2 = np.linspace(.5, .5, 100)
xc2 = zc1 * np.sin(theta)
yc2 = zc1 * np.cos(theta)

zc3 = np.linspace(0, 0, 100)
xc3 = zc1 * np.sin(theta)
yc3 = zc1 * np.cos(theta)


def check(b1,b2,a,b):
	if b1<a:
		b1=a
	#if b2>b:
	#	b2=b
	return b1,b2

def HSV_to_XYZ(H, S, V):
	x = [S*np.cos(H*np.pi/180.0)]
	y = [S*np.sin(H*np.pi/180.0)]
	z = [V]
	return x,y,z

def HSV_to_RGB(H,S,V):
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


def colourr(num):
	if num == 0: #red
		H = np.random.normal(0, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(1, .05, 1)
		name = 'red'
	if num == 1: #green
		H = np.random.normal(120, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(1, .05, 1)
		name = 'green'
	if num == 2: #blue
		H = np.random.normal(240, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(1, .05, 1)
		name = 'blue'
	if num == 3: #yellow
		H = np.random.normal(60, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(1, .05, 1)
		name = 'yellow'
	if num == 4: #aqua
		H = np.random.normal(180, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(1, .05, 1)
		name = 'aqua'
	if num == 5: #black
		H = np.random.normal(0, H_std, 1)
		S = np.random.normal(0, .05, 1)
		V = np.random.normal(0, .05, 1)
		name = 'black'
	if num == 6: #maroon
		H = np.random.normal(0, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(.5, .05, 1)
		name = 'maroon'
	if num == 7: #white
		H = np.random.normal(0, H_std, 1)
		S = np.random.normal(0, .05, 1)
		V = np.random.normal(1, .05, 1)
		name = 'white'
	if num == 8: #silver
		H = np.random.normal(0, H_std, 1)
		S = np.random.normal(0, .05, 1)
		V = np.random.normal(.75, .05, 1)
		name = 'silver'
	if num == 9: #Fuchsia
		H = np.random.normal(300, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(1, .05, 1)
		name = 'Fuchsia'
	if num == 10: #Purple
		H = np.random.normal(300, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(.5, .05, 1)
		name = 'Purple'
	if num == 11: #Olive
		H = np.random.normal(60, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(.5, .05, 1)
		name = 'Olive'


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

	H = int(H)

	R,G,B = HSV_to_RGB(H,S,V)
	n = [R,G,B]
	return [H,S,V],n,name

def distance_calc(A,B):
	return np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

def distance_spatial(dis):
	#if dis <= 2*object_radius:
	#	s = 'touches'
	if dis < 6*object_radius:
		s = 'close to'
	elif dis < 12*object_radius:
		s = 'near'
	else:
		s = 'far from'
	return s

def distance(num):
	if num == 0: #near
		col = np.random.normal(30, 10, 1)
		n = 'r'
		name = 'near'
	if num == 1: #far
		col = np.random.normal(100, 10, 1)
		n = 'b'
		name = 'far from'
	return np.abs(col),n,name

#-----------------------------------------------------------------------------------------------------#
if __name__ == "__main__":


	# Graph stuff
	"""
	f1=plt.figure(1)
	plt.ion()
	plt.show()
	f1.suptitle('World feature graph', fontsize=20)

	f2=plt.figure(2)
	plt.ion()
	plt.show()
	f2.suptitle('Robot hypothesis graph', fontsize=20)
	"""

	plt.ion()
	"""
	f3=plt.figure(3)
	ax3 = f3.add_subplot(111, projection='3d')
	#plt.show()
	f3.suptitle('HSV', fontsize=20)
	ax3.set_xlabel('X')
	ax3.set_ylabel('Y')
	ax3.set_zlabel('Z')
	ax3.plot(xc1, yc1, zc1, c='b')
	ax3.plot(xc2, yc2, zc2, c='b')
	ax3.plot(xc3, yc3, zc3, c='b')
	"""

	f4=plt.figure(4)
	ax4 = f4.add_subplot(111, projection='3d')
	#plt.ion()
	#plt.show()
	f4.suptitle('Robot Hypothesis', fontsize=20)
	ax4.set_xlabel('X')
	ax4.set_ylabel('Y')
	ax4.set_zlabel('Z')
	ax4.plot(xc1, yc1, zc1, c='b')
	ax4.plot(xc2, yc2, zc2, c='b')
	ax4.plot(xc3, yc3, zc3, c='b')

	# ========== Language stuff ========== #
	# inital
	initial2 = ['','this is a ','a ','the ']
	initial = ['','the ']
	iss = ['','is ','is ','is ']


	POINTS = []

	for KK in range(tests):
	# ========== Creating the n_examples ========== #
		examples = {}
		sentence = {}
		H1_all = []				# all the H values
		words = []				# all the words
		words_index = {}			# where did they appear
		hyp = {}

		for i in range(n_examples):
			examples[i] = {}

	#========================== MAIN LOOP ====================================#
		print '============================= Teaching ============================'
		for i in range(n_examples):
			#print number[i],'objects'
			objects = {}
			diss = {}
			obj_number = int(np.random.rand(1)*object_number_max)+1

			#------ object locations ---------#
			# randomly place objects in a map of area 500*500 with object radius being 20
			object_locations = []
			distances = []

			object_locations.append(np.abs(np.random.rand(2)*(map_size-2*object_radius))) 		# first object
		
			for j in range(1,obj_number):
				#while 1:
				obj_2 = np.abs(np.random.rand(2)*(map_size-2*object_radius))
				dis = distance_calc(obj_2,object_locations[j-1])
				distances.append(dis)
				#if dis > 2*object_radius:
				object_locations.append(obj_2)
				#	break
		
			#------ object colours ---------#
			object_colors_names = []
			object_colors = []
			H_value = []
			S_value = []
			V_value = []
		
			H1,color1,name1 = colourr(int(np.random.rand(1)*color_number_max))
			object_colors.append(color1)
			object_colors_names.append(name1)
			H_value.append(H1[0])
			S_value.append(H1[1])
			V_value.append(H1[2])

			#H1_all.append(H1[0])
			for j in range(1,obj_number):
				#while 1:
				H1,color1,name1 = colourr(int(np.random.rand(1)*color_number_max))
				#if name1 not in object_colors_names:
				object_colors.append(color1)
				object_colors_names.append(name1)
				H_value.append(H1[0])
				S_value.append(H1[1])
				V_value.append(H1[2])

			#------- distances ---------#
			object_distances = []
			for j in range(obj_number-1):
				for k in range(j+1,obj_number):
					dis = distance_calc(object_locations[j],object_locations[k])
					distance = distance_spatial(dis)
					diss[str(j)+'-'+str(k)] = dis
					object_distances.append(distance)		

			#------ generating sentences ---------#
			ss = []
			if obj_number==1:
				r1 = int(np.random.rand(1)*4)
				ss.append(initial2[r1]+object_colors_names[0])

			counter = 0			
			for j in range(obj_number):
				for k in range(j+1,obj_number):
					r1 = int(np.random.rand(1)*2)
					r2 = int(np.random.rand(1)*4)
					ss.append(initial[r1]+object_colors_names[j]+' '+iss[r2]+object_distances[counter]+' '+object_colors_names[k])
					counter+=1
			k = ''
			if len(ss)==1:
				sentence[i] = ss[0]
			else:
				for j in range(len(ss)-1):
					k = k+ss[j]+' and '
				k = k+ss[len(ss)-1]
				sentence[i] = k


			#print sentence[i]

			#------ Create a graph structure -----------#
			"""
			o1 = []
			spatial = []

			for p in range(len(H_value)):
				H = str(H_value[p])
				S = str(S_value[p][0])
				V = str(V_value[p][0])
				o1.append([H,S,V])

			for p in distances:
				spatial.append(str(int(p)))

			# actual world graph
			plt.figure(1)
		        plt.cla()
			graph_maker(o1,spatial,plotting)
			plt.axis('off')
		    	plt.draw()
			"""


			#------------- create a list of all words ---------------#
			words2 = []
			for j in sentence[i].split(' '):
				if j not in words:
					words.append(j)
					hyp[j] = {}
					hyp[j]['counter'] = 0	
					hyp[j]['hist_HSV'] = np.zeros(shape=(201,201,101))
					hyp[j]['score_HSV'] = np.zeros(shape=(201,201,101))
					hyp[j]['hist_dist'] = np.zeros(shape=(1000))		# up to 1000 pixel
					hyp[j]['score_dist'] = np.zeros(shape=(1000))		# up to 1000 pixel

				if j not in words2:
					words2.append(j)	

			# -- add 1 to counter for as many times repeated --#
			for j in words2:
				hyp[j]['counter']+=1

			#--------------------------- Compute words histogram for distance -----------------------#
			for j in range(obj_number-1):
				for k in range(j+1,obj_number):
					for jj in words2:
						hyp[jj]['hist_dist'][int(diss[str(j)+'-'+str(k)])] += 1
				
			#============================ TEST THE HYPOTHESIS =================================#
			"""
			Valid_hyp = {}
			counter = 0
			for k in H_value:
				Valid_hyp[counter] = []
				for j in hyp:
					if k in np.nonzero(hyp[j]['score'])[0]:
						Valid_hyp[counter].append(j)
				counter += 1
			
			o1 = []
			for k in Valid_hyp:
				if len(Valid_hyp[k]) == 1:
					o1.append(Valid_hyp[k][0])
				elif len(Valid_hyp[k]) == 0:
					o1.append('none')
				else:
					kk = []
					for jj in Valid_hyp[k]:
						kk.append(jj)
					o1.append(kk)

			# Robot's Hypothesis graph
			plt.figure(2)
		        plt.cla()
			graph_maker(o1,spatial,plotting)
			plt.axis('off')
		    	plt.draw()
			"""

			#------------------ Compute HSV hypothesis ---------------------#

			print '##############################################',words2
			for j in words2:
				hyp[j]['score_HSV'] = np.zeros(shape=(201,201,101))
				for p in range(len(H_value)):
					H = H_value[p]
					S = S_value[p]
					V = V_value[p]
					xs,ys,zs = HSV_to_XYZ(H, S, V)
					x = int(xs[0][0]*100+100)
					y = int(ys[0][0]*100+100)
					z = int(zs[0][0]*100)
					hyp[j]['hist_HSV'][x,y,z] += 1
					c = object_colors[p]
					POINTS.append([xs,ys,zs,c[0]/255.0,c[1]/255.0,c[2]/255.0])

			#-------------------------- Compute HSV Hypotheses ---------------------#
			if i%display_frame == 0:
				for j in words:
					X,Y,Z = np.nonzero(hyp[j]['hist_HSV'])	# loop through the histogram values
					flag = 0
					for kk in [8,12,15,18,20,23,25]:
					    for jj in range(len(X)):
						x1 = X[jj]-kk
						x2 = X[jj]+kk
						y1 = Y[jj]-kk
						y2 = Y[jj]+kk
						z1 = Z[jj]-kk
						z2 = Z[jj]+kk
						x1,x2 = check(x1,x2,0,200)
						y1,y2 = check(y1,y2,0,200)
						z1,z2 = check(z1,z2,0,100)

						# see if a cube in the hitogram can be valid to be a hypothesis
						if  np.sum(hyp[j]['hist_HSV'][x1:x2,y1:y2,z1:z2]) >= hyp[j]['counter']*.9:
							hyp[j]['score_HSV'][x1:x2,y1:y2,z1:z2] = 1
							flag = 1

					    if flag == 1:	# hypothesis has been found
						break

				
				#---------- find which words has hypotheses in HSV ----------#	
				plt.figure(4)
				plt.cla()

				counter = 0
				for j in words:
					A = np.nonzero(hyp[j]['score_HSV'])
					if A[0] != []:
						print j
						x1 = (np.min(A[0])-100)/100.0
						x2 = (np.max(A[0])-100)/100.0
						y1 = (np.min(A[1])-100)/100.0
						y2 = (np.max(A[1])-100)/100.0
						z1 = np.min(A[2])/100.0
						z2 = np.max(A[2])/100.0
						for s, e in combinations(np.array(list(product([x1,x2],[y1,y2],[z1,z2]))), 2):
						    if np.abs(np.sum(np.abs(s-e)) - np.abs(x2-x1))<.01:
							    ax4.plot3D(*zip(s,e), color="gray",alpha=.5)
						    elif np.abs(np.sum(np.abs(s-e)) - np.abs(y2-y1))<.01:
							    ax4.plot3D(*zip(s,e), color="gray",alpha=.5)
						    elif np.abs(np.sum(np.abs(s-e)) - np.abs(z2-z1))<.01:
							    ax4.plot3D(*zip(s,e), color="gray",alpha=.5)
						ax4.text(x1, y1, z1, j, color="black",alpha=.7)
						counter += 1

				#--------------------------- plot histogram for distance -----------------------#
				img_hist = np.zeros(shape=(th*len(hyp),name_window+window, 3), dtype=np.uint8)+255
				img_hist[:,name_window-2:name_window,:] = 0		# name line
				for j in range(len(hyp)):				# horizental lines
					img_hist[j*th:j*th+1,:,:] = 0

				counter = 0
				for j in hyp:
					cv2.putText(img_hist,j+' - '+str(hyp[j]['counter']),(10,(th*(counter+1))-5), cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,0),1)	# write on image
					hist_normalized = hyp[j]['hist_dist'].copy()
					cv2.normalize(hist_normalized,hist_normalized,0,th,cv2.NORM_MINMAX)
					hist_normalized=np.int32(np.around(hist_normalized))
					for x,y in enumerate(hist_normalized):
						cv2.line(img_hist[counter*th:(counter+1)*th,name_window:name_window+window,:],(x,th),(x,th-y),(255,0,0))
					counter += 1

				#============== create a set of believes ===============#
				for j in hyp:
					hyp[j]['score_dist'] = np.zeros(shape=(1000))		# up to 1000 pixel
					flag = 1
					for m in range(30,300,30):
						W = m
						for k in range(window-1):
							if W+k-1 < (window-1):
								if np.sum(hyp[j]['hist_dist'][k:W+k])>=hyp[j]['counter']*.9:
									hyp[j]['score_dist'][k:W+k] = 1
									flag = 0
							else:
								A = np.sum(hyp[j]['hist_dist'][0:-(window-1)+k+W])
								B = np.sum(hyp[j]['hist_dist'][k:(window-1)])
								if A+B>=hyp[j]['counter']*.9:
									hyp[j]['score_dist'][0:-(window-1)+k+W] = 1
									hyp[j]['score_dist'][k:(window-1)] = 1
									flag = 0
						if flag == 0:
							break

				counter = 0
				for j in hyp:
					hist_normalized = hyp[j]['score_dist'].copy()
					cv2.normalize(hist_normalized,hist_normalized,0,th,cv2.NORM_MINMAX)
					hist_normalized=np.int32(np.around(hist_normalized))
					for x,y in enumerate(hist_normalized):
						cv2.line(img_hist[counter*th:(counter+1)*th,name_window:name_window+window,:],(x,th-y+5),(x,th-y),(0,0,255))
					counter += 1
				cv2.imshow('Distance Histogram',img_hist)


				#---------- Plot HSV hypotheses ----------#
				for kk in range(len(POINTS)):
					C = POINTS[kk]
					ax4.scatter(C[0], C[1], C[2], c=(C[3],C[4],C[5]), marker='o')

				ax4.set_xlim([-1,1])
				ax4.set_ylim([-1,1])
				ax4.set_zlim([0,1])

				for kk in range(0,360,rotation_inc):
					plt.figure(4)
					ax4.view_init(elev=25, azim=kk)
				   	plt.draw()
	    				k = cv2.waitKey(time) & 0xFF





			#------ plot object locations ---------#
			img = np.zeros(shape=(map_size, map_size, 3), dtype=np.uint8)+255 
			for j in range(len(object_locations)):
				c = object_colors[j]
				l = object_locations[j]
				cv2.circle(img,(int(l[0])+object_radius,int(l[1])+object_radius),object_radius+2,(0,0,0),-1)
				cv2.circle(img,(int(l[0])+object_radius,int(l[1])+object_radius),object_radius,(c[2],c[1],c[0]),-1)
		

			# draw borders
			img[0:1,:,:] = 0
			img[:,0:1,:] = 0
			img[map_size-1:map_size,:,:] = 0
			img[:,map_size-1:map_size,:] = 0
			#cv2.imwrite(Dir+'Map'+'-'+str(i)+'.png',img)
			cv2.imshow('Map',img)
			cv2.imshow('Distance Histogram',img_hist)
	    		k = cv2.waitKey(time) & 0xFF
			print i+1
	
		print '============================= Results ============================'
		#A = 'obj-'+str(object_number_max)+'-col-'+str(color_number_max)+'-ex-'+str(n_examples)+'-test-'+str(KK)
		#cv2.imwrite(Dir2+'Histogram-'+A+'.png',img_hist)
		#cv2.imwrite(Dir2+'Belief-'+A+'.png',img_b)

