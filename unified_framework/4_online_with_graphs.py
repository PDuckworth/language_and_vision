#! /usr/bin/env python2.7
# -*- coding: iso-8859-1 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import operator
import time
from Graph_functions import *

Dir = '/home/omari/Desktop/Python/language/Simultanous_learning_and_ground/images/'
Dir2 = '/home/omari/Desktop/Python/language/Simultanous_learning_and_ground/images2/'

#============================== Constants ======================================#

object_radius = 20
map_size = 500
object_number_max = 2
n_examples = 2000
frames = 1
H_std = 5
color_number_max = 5
window = 360
name_window = 150
th = 40
tests = 1
plotting = 1
time = 100

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
		name = 'red object'
	if num == 1: #green
		H = np.random.normal(120, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(1, .05, 1)
		name = 'green object'
	if num == 2: #blue
		H = np.random.normal(240, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(1, .05, 1)
		name = 'blue object'
	if num == 3: #yellow
		H = np.random.normal(60, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(1, .05, 1)
		name = 'yellow object'
	if num == 4: #aqua
		H = np.random.normal(180, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(1, .05, 1)
		name = 'aqua object'
	if num == 5: #black
		H = np.random.normal(0, H_std, 1)
		S = np.random.normal(0, .05, 1)
		V = np.random.normal(0, .05, 1)
		name = 'black object'
	if num == 6: #maroon
		H = np.random.normal(0, H_std, 1)
		S = np.random.normal(1, .05, 1)
		V = np.random.normal(.5, .05, 1)
		name = 'maroon object'
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
	
	f1=plt.figure(1)
	plt.ion()
	plt.show()
	f1.suptitle('World feature graph', fontsize=20)

	f2=plt.figure(2)
	plt.ion()
	plt.show()
	f2.suptitle('Robot hypothesis graph', fontsize=20)

	# ========== Language stuff ========== #
	# inital
	initial2 = ['','this is a ','a ','the ']
	initial = ['','the ']
	iss = ['','is ','is ','is ']

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

			H1_all.append(H1[0])
			for j in range(1,obj_number):
				#while 1:
				H1,color1,name1 = colourr(int(np.random.rand(1)*color_number_max))
				#if name1 not in object_colors_names:
				object_colors.append(color1)
				object_colors_names.append(name1)
				H_value.append(H1[0])
				S_value.append(H1[1])
				V_value.append(H1[2])


				H1_all.append(H1[0])
				#	break

			#------- distances ---------#
			object_distances = []
			for j in range(obj_number-1):
				for k in range(j+1,obj_number):
					dis = distance_calc(object_locations[j],object_locations[k])
					distance = distance_spatial(dis)
					#noise = np.random.rand(frames)
					diss[str(j)+'-'+str(k)] = dis
					#ax2.plot(range(frames),diss[str(j)+'-'+str(k)],n)
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


			print sentence[i]

			#------ Create a graph structure -----------#
			#print object_colors_names,object_distances
			#print H_value,distances

			o1 = []
			spatial = []

			for p in H_value:
				o1.append(str(p))

			for p in distances:
				spatial.append(str(int(p)))

			# actual world graph
			plt.figure(1)
		        plt.cla()
			graph_maker(o1,spatial,plotting)
			plt.axis('off')
		    	plt.draw()

			#============================ TEST THE HYPOTHESIS =================================#
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

			#------ create a list of all words ---------#
			words2 = []
			for j in sentence[i].split(' '):
				if j not in words:
					words.append(j)
					hyp[j] = {}
					#hyp[j]['seq'] = []
					hyp[j]['counter'] = 0	
					hyp[j]['hist2'] = np.zeros(360)
					hyp[j]['score'] = np.zeros(360)

				if j not in words2:
					words2.append(j)	

				hyp[j]['hist2'][H_value] += 1 

			# -- add 1 to counter for as many times repeated --#
			for j in words2:
				hyp[j]['counter']+=1

			#------ plot/compute words histogram ---------#
			#if (i+1)%1 == 0:
			img_hist = np.zeros(shape=(th*len(hyp),name_window+window, 3), dtype=np.uint8)+255
			img_hist[:,name_window-2:name_window,:] = 0		# name line
			for j in range(len(hyp)):				# horizental lines
				img_hist[j*th:j*th+1,:,:] = 0

			counter = 0
			for j in hyp:
				cv2.putText(img_hist,j+' - '+str(hyp[j]['counter']),(10,(th*(counter+1))-5), cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,0),1)	# write on image
				hist_normalized = hyp[j]['hist2'].copy()
				cv2.normalize(hist_normalized,hist_normalized,0,th,cv2.NORM_MINMAX)
				hist_normalized=np.int32(np.around(hist_normalized))
				for x,y in enumerate(hist_normalized):
					cv2.line(img_hist[counter*th:(counter+1)*th,name_window:name_window+window,:],(x,th),(x,th-y),(255,0,0))
				counter += 1


		#============== create a set of believes ===============#
			for j in hyp:
				hyp[j]['score'] = np.zeros(360)
				flag = 1
				for m in range(10,60,10):
					W = m
					for k in range(359):
						if W+k-1 < 359:
							if np.sum(hyp[j]['hist2'][k:W+k])>=hyp[j]['counter']*.9:
								hyp[j]['score'][k:W+k] = 1
								flag = 0
						else:
							A = np.sum(hyp[j]['hist2'][0:-359+k+W])
							B = np.sum(hyp[j]['hist2'][k:359])
							if A+B>=hyp[j]['counter']*.9:
								hyp[j]['score'][0:-359+k+W] = 1
								hyp[j]['score'][k:359] = 1
								flag = 0
					if flag == 0:
						#print np.nonzero(hyp[j]['score'])[0]
						#print hyp[j]['hist2'][np.nonzero(hyp[j]['score'])[0]]
						#seq = []
						#for p in range(len(np.nonzero(hyp[j]['score'])[0])):
						#	for m in range(int(hyp[j]['hist2'][np.nonzero(hyp[j]['score'])[0]][p])):
						#		seq.append(np.nonzero(hyp[j]['score'])[0][p])
						#print 'std',np.std(seq)
						#print 'avg',np.mean(seq)
						break
				#print hyp[j]['score']
				#print '----------------------------'

			#print '----'

			#------ plot the set of believes ---------#
			#img_b = np.zeros(shape=(th*len(hyp),name_window+window, 3), dtype=np.uint8)+255
			#img_b[:,name_window-2:name_window,:] = 0		# name line
			#for j in range(len(hyp)):				# horizental lines
			#	img_b[j*th:j*th+1,:,:] = 0

			counter = 0
			for j in hyp:
				#cv2.putText(img_b,j+' - '+str(hyp[j]['counter']),(10,(th*(counter+1))-5), cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,0),1)	# write on image
				hist_normalized = hyp[j]['score'].copy()
				cv2.normalize(hist_normalized,hist_normalized,0,th,cv2.NORM_MINMAX)
				hist_normalized=np.int32(np.around(hist_normalized))
				for x,y in enumerate(hist_normalized):
					cv2.line(img_hist[counter*th:(counter+1)*th,name_window:name_window+window,:],(x,th-y+5),(x,th-y),(0,0,255))
				counter += 1
			cv2.imshow('Histogram',img_hist)
			#cv2.imshow('Belief',img_b)
			
			#------ plot object locations ---------#
			img = np.zeros(shape=(map_size, map_size, 3), dtype=np.uint8)+255 
			for j in range(len(object_locations)):
				c = object_colors[j]
				l = object_locations[j]
				cv2.circle(img,(int(l[0])+object_radius,int(l[1])+object_radius),object_radius,(c[2],c[1],c[0]),-1)
		

			# draw borders
			img[0:1,:,:] = 0
			img[:,0:1,:] = 0
			img[map_size-1:map_size,:,:] = 0
			img[:,map_size-1:map_size,:] = 0
			#cv2.imwrite(Dir+'Map'+'-'+str(i)+'.png',img)
			cv2.imshow('Map',img)
	    		k = cv2.waitKey(time) & 0xFF
			print i+1
	
		print '============================= Results ============================'
		A = 'obj-'+str(object_number_max)+'-col-'+str(color_number_max)+'-ex-'+str(n_examples)+'-test-'+str(KK)
		cv2.imwrite(Dir2+'Histogram-'+A+'.png',img_hist)
		cv2.imwrite(Dir2+'Belief-'+A+'.png',img_b)

