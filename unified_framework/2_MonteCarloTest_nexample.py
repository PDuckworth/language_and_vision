#! /usr/bin/env python2.7
# -*- coding: iso-8859-1 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import operator
import time

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import numpy as np

Dir = '/home/omari/Desktop/Python/language/Simultanous_learning_and_ground/images/'
Dir2 = '/home/omari/Desktop/Python/language/Simultanous_learning_and_ground/images2/'

#============================== Constants ======================================#

object_radius = 20
map_size = 500
object_max = 2
n_examples = 1000
frames = 1
H_std = 10
color_max = 5
window = 360
name_window = 150
th = 40
tests = 100
"""
def online_avg(last_avg, last_N, new_val):
    return ((last_avg*last_N)+new_val)/(last_N+1)

def online_std(last_avg, last_N, last_std, new_val):
    if last_N == 0:
        return 0
    new_avg = online_avg(last_avg, last_N, new_val)
    new_std = ((last_N-1)*last_std + (new_val - last_avg)*(new_val - new_avg))/(last_N)
    #new_std = np.sqrt((last_N*last_std*last_std + (new_val - last_avg)*(new_val - new_avg))/(last_N+1))
    return new_std
"""
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
	if dis <= 2*object_radius:
		s = 'touches'
	elif dis < 6*object_radius:
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


	RESULTS = {}
	# ========== Language stuff ========== #
	# inital
	initial2 = ['','this is a ','a ','the ']
	initial = ['','the ']
	iss = ['','is ','is ','is ']

	object_number_max = object_max
	n_example = [200,500,1000,2000]

	for nnn in n_example:
	  RESULTS[nnn] = {}
	  for color_number_max in range(1,color_max+1):
	    RESULTS[nnn][color_number_max] = 0
	    for KK in range(tests):
	# ========== Creating the n_examples ========== #
		examples = {}
		sentence = {}
		H1_all = []				# all the H values
		words = []				# all the words
		words_index = {}			# where did they appear
		hyp = {}

		for i in range(nnn):
			examples[i] = {}

		#plt.ion()
		#plt.show
		#f, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=False)
	#========================== MAIN LOOP ====================================#
		#print '============================= Teaching ============================'
		for i in range(nnn):
			#print number[i],'objects'
			objects = {}
			diss = {}
			obj_number = int(np.random.rand(1)*object_number_max)+1

			#------ object locations ---------#
			# randomly place objects in a map of area 500*500 with object radius being 20
			object_locations = []

			object_locations.append(np.abs(np.random.rand(2)*(map_size-2*object_radius))) 		# first object
		
			for j in range(1,obj_number):
				#while 1:
				obj_2 = np.abs(np.random.rand(2)*(map_size-2*object_radius))
				dis = distance_calc(obj_2,object_locations[j-1])
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
					hyp[j]['results'] = 0

				if j not in words2:
					words2.append(j)	

				hyp[j]['hist2'][H_value] += 1 

			# -- add 1 to counter for as many times repeated --#
			for j in words2:
				hyp[j]['counter']+=1

			#------ plot/compute words histogram ---------#
			if (i+1)%nnn == 0:
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
				cv2.imshow('Histogram',img_hist)


			#============== create a set of believes ===============#
				for j in hyp:
					hyp[j]['score'] = np.zeros(360)
					for m in range(10,60,10):
						W = m
						for k in range(359):
							if W+k-1 < 359:
								if np.sum(hyp[j]['hist2'][k:W+k])>=hyp[j]['counter']*.9:
									hyp[j]['score'][k:W+k] += 1
									hyp[j]['results'] = 1
							else:
								A = np.sum(hyp[j]['hist2'][0:-359+k+W])
								B = np.sum(hyp[j]['hist2'][k:359])
								if A+B>=hyp[j]['counter']*.9:
									hyp[j]['score'][0:-359+k+W] += 1
									hyp[j]['score'][k:359] += 1
									hyp[j]['results'] = 1
	
					#print hyp[j]['score']
					#print '----------------------------'

				#print '----'

				#------ plot the set of believes ---------#
				img_b = np.zeros(shape=(th*len(hyp),name_window+window, 3), dtype=np.uint8)+255
				img_b[:,name_window-2:name_window,:] = 0		# name line
				for j in range(len(hyp)):				# horizental lines
					img_b[j*th:j*th+1,:,:] = 0

				counter = 0
				for j in hyp:
					cv2.putText(img_b,j+' - '+str(hyp[j]['counter']),(10,(th*(counter+1))-5), cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,0),1)	# write on image
					hist_normalized = hyp[j]['score'].copy()
					cv2.normalize(hist_normalized,hist_normalized,0,th,cv2.NORM_MINMAX)
					hist_normalized=np.int32(np.around(hist_normalized))
					for x,y in enumerate(hist_normalized):
						cv2.line(img_b[counter*th:(counter+1)*th,name_window:name_window+window,:],(x,th),(x,th-y),(255,0,0))
					counter += 1
				cv2.imshow('Belief',img_b)
			
			#------ plot object locations ---------#
			"""
			img = np.zeros(shape=(map_size, map_size, 3), dtype=np.uint8)+255 
			for j in range(len(object_locations)):
				c = object_colors[j]
				l = object_locations[j]
				cv2.circle(img,(int(l[0])+object_radius,int(l[1])+object_radius),object_radius,(c[2],c[1],c[0]),-1)

			#------ saving examples ---------#
			#examples[i]['H_value'] = H_value
			#examples[i]['S_value'] = S_value
			#examples[i]['V_value'] = V_value
			#examples[i]['dis'] = diss

			# draw borders
			img[0:1,:,:] = 0
			img[:,0:1,:] = 0
			img[map_size-1:map_size,:,:] = 0
			img[:,map_size-1:map_size,:] = 0
			#cv2.imwrite(Dir+'Map'+'-'+str(i)+'.png',img)
			cv2.imshow('Map',img)
	    		k = cv2.waitKey(1) & 0xFF
			"""
			#print i+1
	
		#print '============================= Results  ============================'

		A = 0
		for j in hyp:
			A += hyp[j]['results']

		if A == color_number_max:
			print 'Test number =',KK,'Max number of objects =',object_number_max+1,'Max number of colors =',color_number_max ,'result = correct'
			RESULTS[nnn][color_number_max] += 1
		else:
			print 'Test number =',KK,'Max number of objects =',object_number_max+1,'Max number of colors =',color_number_max ,'result = False'
		
		print RESULTS



		A = 'obj-'+str(object_number_max)+'-col-'+str(color_number_max)+'-ex-'+str(n_examples)+'-test-'+str(KK)
		cv2.imwrite(Dir2+'Histogram-'+A+'.png',img_hist)
		cv2.imwrite(Dir2+'Belief-'+A+'.png',img_b)


	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for c, z in zip(['r', 'g', 'b', 'y'], np.arange(0,4)):
	    xs = np.arange(1,color_max+1)
	    ys = []
	    for j in xs:
		ys.append(RESULTS[n_example[z]][j])
	    #ys = np.random.rand(color_max)

	    # You can provide either a single color or an array. To demonstrate this,
	    # the first bar of each set will be colored cyan.
	    cs = [c] * len(xs)
	    #cs[0] = 'c'
	    ax.bar(xs, ys, zs=z+1, zdir='y', color=cs, alpha=0.8)

	ax.set_xlabel('number of colors')
	ax.set_ylabel('number of examples')
	ax.set_zlabel('Results')

	plt.show()



	"""
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)

	x = np.arange(1, color_max+1)
	xs = [.99]
	for j in x:
		xs.append(j)
	xs.append(color_max+.01)

	verts = []
	zs = [0.0]
	for j in range(1,object_max):
		zs.append(j*1.0)



	print 'look',zs,xs
	for z in zs:
	    ys = [0]
	    for x in xs[1:len(xs)-1]:
		    print int(z),int(x),RESULTS[int(z)][int(x)]
		    ys.append(RESULTS[int(z)][int(x)])
	    ys.append(0)
	    verts.append(list(zip(xs, ys)))

	poly = PolyCollection(verts, facecolors = [cc('r'), cc('g'), cc('b'),
		                                   cc('y')])
	poly.set_alpha(0.7)

	for j in range(len(zs)):
		zs[j]+=1

	ax.add_collection3d(poly, zs=zs, zdir='y')

	ax.set_xlabel('Number of colors')
	ax.set_xlim3d(0, object_max+1)
	ax.set_ylabel('Number of objects')
	ax.set_ylim3d(0, color_max+1)
	ax.set_zlabel('results')
	ax.set_zlim3d(0, tests)

	plt.show()
	"""












