#! /usr/bin/env python2.7
# -*- coding: iso-8859-1 -*-

#3DMouse (C) Josh Wedlake 2012
#released under the terms of the GPLv3

import datetime
import numpy as np
import cv2
import colorsys
import math

#------------setup-----------

cam_sep=0.1
use_hsv=False
sq_tol=0.5 #tolerance on what is considered a square
diff_levels=3 #prediction based on differentials, default cubic
mirror_mode=True
use_chessboard=True
use_advanced_calib=False
calib_count=25

#----------------------------


video_wnd_name="Tracker"
drawing_wnd_name="3D Drawing"

#possible remappings
prs=(
	(0,1,2,3),
	(1,2,3,0),
	(2,3,0,1),
	(3,0,1,2),
	(3,2,1,0),
	(2,1,0,3),
	(1,0,3,2),
	(0,3,2,1),
	(3,2,1,0)	
	)

#3d square space
square_3D=np.array(((-0.5,-0.5,0.),(0.5,-0.5,0.),(0.5,0.5,0.),(-0.5,0.5,0.)))

chessboard_3D=np.array((
	(-0.5,-0.5,0.),(0.,-0.5,0.),(0.5,-0.5,0.),
	(-0.5,0.,0.),(0.,0.,0.),(0.5,0.,0.),
	(-0.5,0.5,0.),(0.,0.5,0.),(0.5,0.5,0.)
	))


#cam distortion
cam_dist=np.array(((0.,0.,0.,0.),))


def reset_camera(cam_sep):
	if mirror_mode:
		cam_lvec=np.array([[[cam_sep,0.,0.]]])
		cam_rvec=np.array([[[-cam_sep,0.,0.]]])
	else:
		cam_lvec=np.array([[[-cam_sep,0.,0.]]])
		cam_rvec=np.array([[[cam_sep,0.,0.]]])
	cam_rmat=np.matrix([[1.,0.,0.],
		[0.,1.,0.],
		[0.,0.,1.]
		])
	return cam_lvec,cam_rvec,cam_rmat

	
def angle_cos(p0, p1, p2):
	d1, d2 = p0-p1, p2-p1
	return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img,sq_tol):
	img = cv2.GaussianBlur(img, (5, 5), 0)
	squares = []
	for gray in cv2.split(img):
		for thrs in xrange(0, 255, 26):
			if thrs == 0:
				bin = cv2.Canny(gray, 0, 50, apertureSize=5)
				bin = cv2.dilate(bin, None)
			else:
				retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
			contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			for cnt in contours:
				cnt_len = cv2.arcLength(cnt, True)
				cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
				if len(cnt) == 4 and test_edge(cnt,img.shape) and 30000 > cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
					cnt = cnt.reshape(-1, 2)
					max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
					#0.3 is default, 0.5 is more tolerant
					if max_cos < sq_tol:
						squares.append(cnt)
	return squares

def points_to_contour(in_points):
	pc=len(in_points)
	out_cont=np.ndarray(shape=(pc,1,2), dtype="Float32")
	for tp in range(0,pc):
		out_cont[tp,0,0]=in_points[tp,0]
		out_cont[tp,0,1]=in_points[tp,1]
	return out_cont

def test_edge(test_cont,img_shape):
	#returns true if the square doesn't touch the screen edge
	for i in xrange(4):
		if test_cont[i,0,0]<3 or test_cont[i,0,0]>(img_shape[1]-3) or test_cont[i,0,1]<3 or test_cont[i,0,1]>(img_shape[0]-3):
			#print(test_cont)
			return False
			#return True
	return True
	
def is_square(test_square,sq_tol,img_shape):
	temp_cont=points_to_contour(test_square)
	if test_edge(temp_cont,img_shape) and 30000 > cv2.contourArea(temp_cont) > 1000 and cv2.isContourConvex(temp_cont):
		max_cos = np.max([angle_cos( test_square[i], test_square[(i+1) % 4], test_square[(i+2) % 4] ) for i in xrange(4)])
		if max_cos < sq_tol:
			return True
	return False
	
	
def adapt_calib_color(img,this_square,min_color,max_color,mean_color,sd_color):
	#find square centre
	point_sum=np.array((0,0))
	for each_point in this_square:
		point_sum+=each_point
	point_sum/=4
	#take a slice of the image
	img_slice=img[int(point_sum[1]-2):int(point_sum[1]+2),int(point_sum[0]-2):int(point_sum[0]+2)]
	if(use_hsv):
		img_slice_HSV=cv2.cvtColor(img_slice,cv2.COLOR_BGR2HSV)
		mean=cv2.mean(img_slice_HSV)
	else:
		mean=cv2.mean(img_slice)
	if len(mean)==3:
		min_color=(min_color*0.9)+((mean-sd_color)*0.1)
		max_color=(max_color*0.9)+((mean+sd_color)*0.1)
		if(use_hsv):
			return min_color,max_color,mean_color_in_rgb(mean)
		else:
			return min_color,max_color,(mean[0,0], mean[1,0], mean[2,0])
	else:
		return min_color,max_color,mean_color

def mean_color_in_rgb(mean):
	mc_temp=colorsys.hsv_to_rgb(mean[0,0]/255, mean[1,0]/255, mean[2,0]/255)
	mean_color=(mc_temp[2]*255,mc_temp[1]*255,mc_temp[0]*255)
	return mean_color

def setup_camera(capture,cam_dist):
	if use_advanced_calib:
		calib_corners=np.array(())
		points_2D=[]
		points_3D=[]
		pattern_size = (3, 3)
		pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
		pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
		pattern_points*=0.1
		img_count=0
		while True:
			cap_success,img=capture.read()
			if cap_success:
				corner_success,calib_corners=cv2.findChessboardCorners(img, (3,3), calib_corners, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
				if corner_success:
					if img_count==0:
						img_draw=np.zeros_like(img)
					cv2.cornerSubPix(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), calib_corners, (11, 11), (-1, -1),(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,20, 0.03))
					points_2D.append(calib_corners.reshape(-1, 2))
					points_3D.append(pattern_points)
					img_count+=1
				#mirror the image
				if mirror_mode:
					img=cv2.flip(img,1)
				cv2.putText(img,"CALIBRATING "+str(img_count)+"/"+str(calib_count),(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
				cv2.imshow(video_wnd_name,img)
				if img_count==calib_count:
					h, w = img.shape[:2]
					cam_mat = np.zeros((3, 3))
					c_success,cam_mat,cam_dist,rvecs,tvecs=cv2.calibrateCamera(points_3D,points_2D, (w, h),cam_mat,cam_dist)
					return True,cam_mat,cam_dist,img_draw
			key=cv2.waitKey(10)
			if key==27:
				return False,0,0,0
	else:
		cap_success=False
		while cap_success==False:
			cap_success,img=capture.read()
			if cap_success:
				cam_mat=guess_camera_matrix(img)
				img_draw=np.zeros_like(img)
				return True,cam_mat,cam_dist,img_draw
			key=cv2.waitKey(10)
			if key==27:
				return False,0,0,0
	
def calib_color(capture):
	#calibration mode
	patch_size=40
	while True:
		#get the frame
		cap_success,img=capture.read()
		#if theres an image process it
		if(cap_success):
			#guess the matrix first
			cam_mat=guess_camera_matrix(img)
			img_draw=np.zeros_like(img)
			midpoint=(img.shape[0]/2,img.shape[1]/2)
			points=((midpoint[1]-patch_size,midpoint[0]-patch_size),
				(midpoint[1]-patch_size,midpoint[0]+patch_size),
				(midpoint[1]+patch_size,midpoint[0]+patch_size),
				(midpoint[1]+patch_size,midpoint[0]-patch_size))
			key=cv2.waitKey(10)
			if key==99:
				#get the average hsv within the rect
				cropimg=img[int(midpoint[0]-patch_size):int(midpoint[0]+patch_size),int(midpoint[1]-patch_size):int(midpoint[1]+patch_size)]
				if(use_hsv):
					cropimg=cv2.cvtColor(cropimg,cv2.COLOR_BGR2HSV)
				mean,sd=cv2.meanStdDev(cropimg)
				sd*=10
				min_color=mean-sd
				max_color=mean+sd
				if(use_hsv):
					return True,min_color,max_color,mean_color_in_rgb(mean),sd,cam_mat,img_draw
				else:
					return True,min_color,max_color,(mean[0,0], mean[1,0], mean[2,0]),sd,cam_mat,img_draw
			elif key==91:
				patch_size-=1
			elif key==93:
				patch_size+=1
			elif key==27:
				return False,0,0,0,0,0,0
			cv2.drawContours(img, np.array((
				(points[0],points[1]),
				(points[1],points[2]),
				(points[2],points[3]),
				(points[3],points[0])
				)), -1, (0, 255, 0), 3 )
			#mirror the image
			if mirror_mode:
				img=cv2.flip(img,1)
			cv2.putText(img,"C to calibrate, ESC to escape, [/] to change patch size",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
			cv2.imshow(video_wnd_name,img)
		elif cv2.waitKey(10)==27:
			return False,0,0,0,0,0,0
			
def get_thresholded_image(img,min_color,max_color):
	#convert to HSV
	if(use_hsv):
		imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		imgThresh=cv2.inRange(imgHSV,min_color,max_color)
	else:
		imgThresh=cv2.inRange(img,min_color,max_color)
	return imgThresh

def get_square_distance(this_square,last_square):
	sum_dist=0
	for point_ref in range(0,4):
		p_dist=((this_square[point_ref][0]-last_square[point_ref][0])**2)+((this_square[point_ref][1]-last_square[point_ref][1])**2)
		sum_dist+=p_dist
	return sum_dist
	
def get_edge_length_sum(this_square):
	sum_dist=0
	sum_dist+=(((this_square[1][0]-this_square[0][0])**2)+((this_square[1][1]-this_square[0][1])**2))**0.5
	sum_dist+=(((this_square[2][0]-this_square[1][0])**2)+((this_square[2][1]-this_square[1][1])**2))**0.5
	sum_dist+=(((this_square[3][0]-this_square[2][0])**2)+((this_square[3][1]-this_square[2][1])**2))**0.5
	sum_dist+=(((this_square[0][0]-this_square[3][0])**2)+((this_square[0][1]-this_square[3][1])**2))**0.5
	return sum_dist

def reset_diff_data(first_rect,levels):
	diff_data=[[first_rect,first_rect]]
	for tl in range(0,levels):
		diff_data.append([np.zeros(shape=(4,2),dtype="Float32"),np.zeros(shape=(4,2),dtype="Float32")])
	return diff_data

def rect_motion_differential(diff_data,this_square,levels):
	'''
	diff data is a set of levels
	[thisrect,lastrect]
	[this1dif,last1dif]
	[this2dif,last2dif]
	etc
	'''
	
	smooth_data=0.9
	invs=1-smooth_data
	
	if len(diff_data)==0:
		diff_data=reset_diff_data(this_square,levels)
		return diff_data
	else:
		diff_data[0][1]=diff_data[0][0]
		diff_data[0][0]=this_square
		for level_id in range(1,len(diff_data)):
			diff_data[level_id][1]=diff_data[level_id][0]
			diff_data[level_id][0]=((diff_data[level_id-1][0]-diff_data[level_id-1][1])*invs)+(diff_data[level_id][1]*smooth_data)
		return diff_data
	
def predict_next_rect(diff_data):
	next=diff_data[len(diff_data)-1][0]-diff_data[len(diff_data)-1][1]
	for inv_level in range(0,len(diff_data)):
		this_level=(len(diff_data)-1)-inv_level
		next+=diff_data[this_level][0]
	return next
	
def guess_camera_matrix(img):
	#c=(img.shape[0]/2,img.shape[1]/2)
	c=(img.shape[0]/2,img.shape[1]/2)
	#guess?
	f=(img.shape[0],img.shape[0])
	return np.array(((f[0],0.,c[0]),
		(0.,f[1],c[1]),
		(0.,0.,0.)))
	
def find_likely_square(squares,last_square,img_last,img_this,img_last_col,img_this_col,sq_tol):
	#last square is an array of points 1,2,3,4
	#returns (-1 for fail 0 for success 1 for success but don't adapt color,square)
	most_probable=[-1,[]]
	if len(squares)>0:
		if len(last_square)!=0:
			min_dist=0
			for each_square in squares:
				#try mapping last square onto square
				for er in prs:
					poss_square=np.array((each_square[er[0]],each_square[er[1]],each_square[er[2]],each_square[er[3]]))
					#find the sum point to point distance
					this_dist=get_square_distance(poss_square,last_square)
					#if its min then save it
					if most_probable[0]==-1 or (most_probable[0]!=-1 and this_dist<min_dist):
						min_dist=this_dist
						most_probable=[0,np.array(poss_square)]
		else:
			#return the smallest of the squares
			min_sum=0
			for each_square in squares:
				this_sum=get_edge_length_sum(each_square)
				if most_probable[0]==-1 or (most_probable[0]!=-1 and this_sum<min_sum):
					min_sum=this_sum
					most_probable=[0,np.array(each_square)]
	else:
		#motion track the features if a last image is available
		if(len(img_last)!=0 and len(last_square)!=0):
			#print("square not found, trying optical flow")
			temp_points=np.ndarray(shape=(4,1,2), dtype=float)
			for tp in range(0,4):
				temp_points[tp,0,0]=last_square[tp,0]
				temp_points[tp,0,1]=last_square[tp,1]
			temp_points=temp_points.astype("float32")
			temp_result=cv2.calcOpticalFlowPyrLK(img_last, img_this, temp_points)
			temp_square=np.array((
				(temp_result[0][0][0][0],temp_result[0][0][0][1]),
				(temp_result[0][1][0][0],temp_result[0][1][0][1]),
				(temp_result[0][2][0][0],temp_result[0][2][0][1]),
				(temp_result[0][3][0][0],temp_result[0][3][0][1]),
				))
			
			#check which points were found
			average_vec=np.array((0,0))
			found_count=0
			for tp in range(0,4):
				if temp_result[1][tp][0]==1:
					average_vec+=(temp_result[0][tp][0]-last_square[tp])
					found_count+=1
			
			
			if(found_count==0) and len(img_last_col)>0:
				#try using colours
				#print("square not found - trying full color motion track")
				#find some features
				track_features=cv2.goodFeaturesToTrack(cv2.cvtColor(img_last_col,cv2.COLOR_BGR2GRAY), 25,0.05,10)
				#track them
				temp_result=cv2.calcOpticalFlowPyrLK(img_last_col, img_this_col, track_features)
				#next square space
				temp_square=np.zeros(shape=(4,2),dtype="Float32")
				for point_index in range(0,4):
					each_point=last_square[point_index]
					closest_result=-1
					min_dist=0
					for result_index in range(0,len(temp_result[1])):
						if temp_result[1][result_index][0]==1:
							vec_diff=track_features[result_index][0]-each_point
							this_dist=np.sqrt(np.vdot(vec_diff,vec_diff))
							if closest_result==-1 or (closest_result!=-1 and this_dist<min_dist):
								closest_result=result_index
								point_transform=temp_result[0][result_index][0]-track_features[result_index][0]
					
					#if nothing was found, give up
					if closest_result==-1:
						break
					else:
						temp_square[point_index]=last_square[point_index]+point_transform
				
				if point_index==3 and closest_result!=-1:
					if(is_square(temp_square,sq_tol,img_this.shape)):
						most_probable[1]=temp_square
						most_probable[0]=1 # don't adapt color
					
			elif(found_count==4):
				#print("4 points found")
				if(is_square(temp_square,sq_tol,img_this.shape)):
					most_probable[1]=temp_square
					most_probable[0]=0
			else:
				#print("partial square found")
				average_vec/=found_count
				for tp in range(0,4):
					if temp_result[1][tp][0]==0:
						temp_square[tp]=last_square[tp]+average_vec
				if(is_square(temp_square,sq_tol,img_this.shape)):
					most_probable[1]=temp_square
					most_probable[0]=0
		#else:
		#	print("not enough information to predict a square")
	#if(most_probable[0]==-1):
	#	print("FAIL")
	return most_probable

	
def translation_matrix(direction):
	M = np.identity(4)
	M[:3, 3] = direction[:3]
	return M

def translation_from_matrix(matrix):
	return np.array(matrix, copy=False)[:3, 3].copy()
	
def euler_matrix(angle):
	si, sj, sk = math.sin(angle[0]), math.sin(angle[1]), math.sin(angle[2])
	ci, cj, ck = math.cos(angle[0]), math.cos(angle[1]), math.cos(angle[2])
	cc, cs = ci*ck, ci*sk
	sc, ss = si*ck, si*sk
	
	M = np.identity(3)
	
	M[0,0] = cj*ck
	M[0,1] = sj*sc-cs
	M[0,2] = sj*cc+ss
	M[1,0] = cj*sk
	M[1,1] = sj*ss+cc
	M[1,2] = sj*cs-sc
	M[2,0] = -sj
	M[2,1] = cj*si
	M[2,2] = cj*ci
	
	return M
	
def process_3D_data(this_square,last_square,last_point,square_3D,cam_mat,cam_dist,average_point,point_count):
	#stabilise slightly
	if(len(last_square)>0):
		analyse_square=(this_square.astype("Float32")+last_square.astype("Float32"))/2
	else:
		analyse_square=this_square.astype("Float32")
	#last transform contains rvec,tvec - use them for stabilisation
	PnPsuccess,rvec,tvec=cv2.solvePnP(square_3D, analyse_square, cam_mat,cam_dist)
	if(PnPsuccess):
		this_point=tvec
		point_count+=1
		
		if len(last_point)!=0:
			this_point=(this_point+last_point)/2
			average_point=(this_point*(1/float(point_count)))+(average_point*((point_count-1.)/float(point_count)))
			return this_point,average_point,point_count
		else:
			average_point=(this_point*(1/float(point_count)))+(average_point*((point_count-1.)/float(point_count)))
			return this_point,average_point,point_count
	else:
		if len(last_point)!=0:
			return last_point,average_point,point_count
		else:
			return np.array((0.,0.,0.)),average_point,point_count

def translate_camera(cam_lvec,cam_rvec,cam_rmat,translation):
	'''
	xvec=(translation[0]*cam_rmat[0]).tolist()[0]
	yvec=(translation[1]*cam_rmat[1]).tolist()[0]
	zvec=(translation[2]*cam_rmat[2]).tolist()[0]
	'''
	xvec=np.array([translation[0]*cam_rmat[0]])
	yvec=np.array([translation[1]*cam_rmat[1]])
	zvec=np.array([translation[2]*cam_rmat[2]])
	tvec=xvec+yvec+zvec
	print(tvec,cam_lvec)
	cam_lvec+=tvec
	cam_rvec+=tvec
	return cam_lvec,cam_rvec


def rotate_camera_around_cursor(cam_lvec,cam_rvec,cam_rmat,centre_point,rotation):
	
	cam_transform=np.float64([[[centre_point[0][0],centre_point[1][0],centre_point[2][0]]]])
	cam_lvec-=cam_transform
	cam_rvec-=cam_transform
	cam_lvec=cv2.transform(cam_lvec, euler_matrix(rotation))
	cam_rvec=cv2.transform(cam_rvec, euler_matrix(rotation))
	cam_lvec+=cam_transform
	cam_rvec+=cam_transform
	
	#rotation
	
	#cam_rmat=euler_matrix(rotation)*cam_rmat
	cam_rmat=euler_matrix(rotation)*cam_rmat
	
	return cam_lvec,cam_rvec,cam_rmat

def re_draw(lines,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec):
	img_draw=np.zeros_like(img_draw)
	cb,cg,cr=cv2.split(img_draw)
	for each_line in lines:
		if len(each_line)>3:
			#red view
			points_screen,jacobian=cv2.projectPoints(np.array(each_line), cam_rmat, cam_lvec, cam_mat, cam_dist)
			contours=np.array((points_screen.reshape(-1, 2),)).astype(int)
			cv2.polylines(cr, contours, False, (255),1)
			#blue view
			points_screen,jacobian=cv2.projectPoints(np.array(each_line), cam_rmat, cam_rvec, cam_mat, cam_dist)
			contours=np.array((points_screen.reshape(-1, 2),)).astype(int)
			cv2.polylines(cb, contours, False, (255),1)
	img_draw=cv2.merge((cb,cb,cr))

	if mirror_mode:
		cv2.imshow(drawing_wnd_name,cv2.flip(img_draw,1))
	else:
		cv2.imshow(drawing_wnd_name,img_draw)
	return img_draw

def save_obj_file(lines):
	verts=""
	edges=""
	file="o Drawing\n"
	sum_point_index=0
	for each_line in lines:
		for point_index,each_point in enumerate(each_line):
			verts+="v "+str(each_point[0][0])+" "+str(each_point[1][0])+" "+str(each_point[2][0])+"\n"
			sum_point_index+=1
			if(point_index>0):
				edges+="f "+str(sum_point_index)+" "+str(sum_point_index-1)+"\n"
			
	file+=verts+edges
	with open('out'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'.obj','w') as output_file:
		output_file.write(file)
        output_file.close()

		
def simple_draw(last_point,this_point,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec):
	if len(last_point)!=0:
		cb,cg,cr=cv2.split(img_draw)
		#red view
		points_screen,jacobian=cv2.projectPoints(np.array((last_point,this_point)), cam_rmat, cam_lvec, cam_mat, cam_dist)
		contours=np.array((((points_screen [0,0,0],points_screen [0,0,1]),(points_screen [1,0,0],points_screen [1,0,1])),)).astype(int)
		cv2.polylines(cr, contours, False, (255),1)
		#blue view
		points_screen,jacobian=cv2.projectPoints(np.array((last_point,this_point)), cam_rmat, cam_rvec, cam_mat, cam_dist)
		contours=np.array((((points_screen [0,0,0],points_screen [0,0,1]),(points_screen [1,0,0],points_screen [1,0,1])),)).astype(int)
		cv2.polylines(cb, contours, False, (255),1)
		img_draw=cv2.merge((cb,cb,cr))
		if mirror_mode:
			cv2.imshow(drawing_wnd_name,cv2.flip(img_draw,1))
		else:
			cv2.imshow(drawing_wnd_name,img_draw)
		return img_draw
	return img_draw


	
if __name__ == "__main__":
	#default view
	cam_lvec,cam_rvec,cam_rmat=reset_camera(cam_sep)
	#setup webcam
	capture = cv2.VideoCapture(0)
	cv2.namedWindow(video_wnd_name, 1)
	cv2.namedWindow(drawing_wnd_name, 1)
	#calib
	if use_chessboard:
		mean_color=(255,255,255)
		corners=np.array(())
		calib_success,cam_mat,cam_dist,img_draw=setup_camera(capture,cam_dist)
	else:
		calib_success,orig_min_color,orig_max_color,orig_mean_color,sd_color,cam_mat,img_draw=calib_color(capture)
		min_color,max_color,mean_color=orig_min_color,orig_max_color,orig_mean_color
		
		
	
	if calib_success:
		viewing_only=False
		last_square=[]
		last_point=[]
		average_point=np.array((0.,0.,0.))
		point_count=0
		img_last=[]
		img_last_col=[]
		diff_data=[]
		lines=[[]]
		consecutive_fails=0
		while True:
			#get the frame
			cap_success,img_col=capture.read()
			#if theres an image process it        
			if(cap_success):
						
				status_text=""
				
				if not viewing_only:
					if use_chessboard:
						img=img_col
						corner_success,corners=cv2.findChessboardCorners(img_col, (3,3), corners, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK) 
						if corner_success:
							likely_square=np.array((corners[0][0],corners[2][0],corners[8][0],corners[6][0]))
							cv2.cornerSubPix(cv2.cvtColor(img_col,cv2.COLOR_BGR2GRAY), likely_square, (11, 11), (-1, -1),(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,20, 0.03))
							if(cv2.isContourConvex(points_to_contour(likely_square))):
								squares=[likely_square]
							else:
								squares=[]
						else:
							squares=[]
						search_success,likely_square=find_likely_square(squares,last_square,img_last,img,img_last_col,img_col,sq_tol)
					else:
						img=get_thresholded_image(img_col,min_color,max_color)
						img=cv2.convertScaleAbs(img)
						squares = find_squares(img,sq_tol)
						search_success,likely_square=find_likely_square(squares,last_square,img_last,img,img_last_col,img_col,sq_tol)
						img_last=img
						#convert back to color for display
						img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
						
					img_last_col=img_col
					
					if(search_success!=-1):
						status_text="FOUND"
					
						#find the 3D point
						this_point,average_point,point_count=process_3D_data(likely_square,last_square,last_point,square_3D,cam_mat,cam_dist,average_point,point_count)
						img_draw=simple_draw(last_point,this_point,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec)
						last_point=this_point
						lines[len(lines)-1].append(this_point)
						
						#reset the consecutive fails
						consecutive_fails=0
						#adapt color if the search is not optical flow based
						if search_success==0 and not use_chessboard:
							min_color,max_color,mean_color=adapt_calib_color(img_col,likely_square,min_color,max_color,mean_color,sd_color)
						else:
							#min_color,max_color=orig_min_color,orig_max_color
							consecutive_fails+=0.5
						cv2.drawContours(img, np.array((likely_square,)).astype(int), -1, (0,0,0), 5 )
						cv2.drawContours(img, np.array((likely_square,)).astype(int), -1, mean_color, 3 )
						
						#update differentials
						diff_data=rect_motion_differential(diff_data,likely_square,diff_levels)
						last_square=likely_square
					elif consecutive_fails<1 and len(diff_data)>0:
						status_text="Guessing"
						
						#predict
						likely_square=predict_next_rect(diff_data)
						
						#find the 3D point
						this_point,average_point,point_count=process_3D_data(likely_square,last_square,last_point,square_3D,cam_mat,cam_dist,average_point,point_count)
						img_draw=simple_draw(last_point,this_point,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec)
						last_point=this_point
						lines[len(lines)-1].append(this_point)
						
						cv2.drawContours(img, np.array((likely_square,)).astype(int), -1, (0,0,0), 5 )
						cv2.drawContours(img, np.array((likely_square,)).astype(int), -1, mean_color, 3 )
						
						#update differentials
						diff_data=rect_motion_differential(diff_data,likely_square,diff_levels)
						consecutive_fails+=1
					else:
						status_text="Searching"
						
						lines.append([])
						last_point=[]
						
						if not use_chessboard:
							#reset the adpatation colour
							min_color,max_color=orig_min_color,orig_max_color
							#reset the last square
							last_square=[]
				else:
					status_text="Viewing"
					img=img_col
				#draw the tracking window
				if mirror_mode:
					img=cv2.flip(img,1)
				cv2.putText(img,"I/J/K/M - translate view",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,mean_color,1)
				cv2.putText(img,"W/A/S/Z - rotate view",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,mean_color,1)
				cv2.putText(img,",/. - adjust tolerance",(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,mean_color,1)
				cv2.putText(img,"R - reset view",(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,mean_color,1)
				cv2.putText(img,"N - new",(20,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,mean_color,1)
				cv2.putText(img,"O - write OBJ file",(20,120),cv2.FONT_HERSHEY_SIMPLEX,0.5,mean_color,1)
				cv2.putText(img,"C - recalibrate",(20,140),cv2.FONT_HERSHEY_SIMPLEX,0.5,mean_color,1)
				cv2.putText(img,"V - view/draw",(20,160),cv2.FONT_HERSHEY_SIMPLEX,0.5,mean_color,1)
				cv2.putText(img,status_text,(20,180),cv2.FONT_HERSHEY_SIMPLEX,0.5,mean_color,1)
				cv2.imshow(video_wnd_name,img)

			key=cv2.waitKey(10)
			
			if key==27:
				lines.append([])
				break
			elif key==111:
				#save the points to a file
				save_obj_file(lines)
			elif key==110:
				#start over
				last_point=[]
				average_point=np.array((0.,0.,0.))
				point_count=0
				last_square=[]
				diff_data=[]
				lines=[[]]
				img_draw=re_draw(lines,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec)
			elif key==44:
				#reduce tolerance
				sd_color*=0.9
			elif key==46:
				#increase tolerance
				sd_color*=1.1
			elif key==114:
				#reset view
				cam_lvec,cam_rvec,cam_rmat=reset_camera(cam_sep)
				img_draw=re_draw(lines,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec)
			elif key==118:
				#viewing mode only - V
				if viewing_only:
					viewing_only=False
				else:
					viewing_only=True
			#rotate view
			elif key==97:
				cam_lvec,cam_rvec,cam_rmat=rotate_camera_around_cursor(cam_lvec,cam_rvec,cam_rmat,average_point,np.array((0,0.2,0)))
				img_draw=re_draw(lines,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec)
			elif key==115:
				cam_lvec,cam_rvec,cam_rmat=rotate_camera_around_cursor(cam_lvec,cam_rvec,cam_rmat,average_point,np.array((0,-0.2,0)))
				img_draw=re_draw(lines,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec)
			elif key==119:
				cam_lvec,cam_rvec,cam_rmat=rotate_camera_around_cursor(cam_lvec,cam_rvec,cam_rmat,average_point,np.array((0.2,0,0)))
				img_draw=re_draw(lines,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec)
			elif key==122:
				cam_lvec,cam_rvec,cam_rmat=rotate_camera_around_cursor(cam_lvec,cam_rvec,cam_rmat,average_point,np.array((-0.2,0,0)))
				img_draw=re_draw(lines,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec)
			#translate view
			elif key==105:
				cam_lvec,cam_rvec=translate_camera(cam_lvec,cam_rvec,cam_rmat,np.array((0,0,-1)))
				img_draw=re_draw(lines,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec)
			elif key==107:
				cam_lvec,cam_rvec=translate_camera(cam_lvec,cam_rvec,cam_rmat,np.array((1,0,0)))
				img_draw=re_draw(lines,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec)
			elif key==106:
				cam_lvec,cam_rvec=translate_camera(cam_lvec,cam_rvec,cam_rmat,np.array((-1,0,0)))
				img_draw=re_draw(lines,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec)
			elif key==109:
				cam_lvec,cam_rvec=translate_camera(cam_lvec,cam_rvec,cam_rmat,np.array((0,0,1)))
				img_draw=re_draw(lines,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec)
			elif key==99:
				if use_chessboard:
					calib_success,cam_mat,cam_dist,img_draw=setup_camera(capture,cam_dist)
					img_draw=re_draw(lines,img_draw,cam_lvec,cam_rmat,cam_mat,cam_dist,cam_rvec)
					if(not calib_success):
						break
				else:
					calib_success,orig_min_color,orig_max_color,orig_mean_color,sd_color,cam_mat,img_draw=calib_color(capture)
					if(not calib_success):
						break
					else:
						min_color,max_color,mean_color=orig_min_color,orig_max_color,orig_mean_color
						diff_data=[]
		
	cv2.destroyAllWindows()
