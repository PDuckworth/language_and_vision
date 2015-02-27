
import numpy as np
import cv2

object_radius = 20	#used in Scene_functions and QSR_functions

Dir = '/home/omari/Desktop/Python/language/Simultanous_learning_and_ground/images/'
Dir2 = '/home/omari/Desktop/Python/language/Simultanous_learning_and_ground/images2/'

#-------------------------------------------------------------------------------------#
def object_qsr(o_loc):
	o_qsr = {}
	obj_number = len(o_loc)
	o_qsr['obj_number']=obj_number
	for j in range(obj_number-1):
		for k in range(j+1,obj_number):
			# distance
			dis = distance_calc(o_loc[j][0],o_loc[k][0])
			distance = distance_spatial(dis)
			o_qsr[str(j)+'-'+str(k)+'-dis'] = int(dis*100)/100.0
			o_qsr[str(j)+'-'+str(k)+'-spa'] = distance

			# direciton			
			direction,angle = direction_calc(o_loc[j][0],o_loc[k][0])	# referenced to the first object
			o_qsr[str(j)+'-'+str(k)+'-ang'] = int(angle*100)/100.0
			o_qsr[str(j)+'-'+str(k)+'-dir'] = direction
	#o_dir = 1
	return o_qsr

#-------------------------------------------------------------------------------------#
def direction_calc(A,B):
	X = A[0]-B[0]
	Y = A[1]-B[1]
	C = np.round(np.arctan2(Y,X)/np.pi*4)/4.0

	if C == 1 or C == -1:
		d = 'to the left of'
	elif C == -0 or C == 0.0:
		d = 'to the right of'  
	elif C == -0.75:
		d = 'to the top_left of'
	elif C == 0.75:
		d = 'to the bottom_left of'
	elif C == -0.5:
		d = 'above of'
	elif C == 0.5:
		d = 'below of'
	elif C == -0.25:
		d = 'to the top_right of'
	elif C == 0.25:
		d = 'to the bottom_right of'

	return d,np.arctan2(Y,X)

#-------------------------------------------------------------------------------------#
def distance_calc(A,B):
	return np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

#-------------------------------------------------------------------------------------#
def distance_spatial(dis):
	if dis <= 2*object_radius:
		s = 'touches'
	if dis < 6*object_radius:
		s = 'close to'
	elif dis < 11*object_radius:
		s = 'near'
	elif dis < 16*object_radius:
		s = 'far from'
	else:
		s = 'very_far from'
	return s
