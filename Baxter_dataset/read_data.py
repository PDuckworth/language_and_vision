import numpy as np
import cv2
import getpass
import os, os.path
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal

class read_data():
    """docstring for read_data"""
    def __init__(self):
        # ****************************************************************************************************
        self.dir = '/home/' + getpass.getuser() + '/Datasets/scene'
        self.scene = 2
        self.frame = 1
        self.get_frame_str()
        self.update_scene(self.scene)
        self.o_data = {}            # object tracks
        self.r_data = {}            # robot tracks

    def update_scene(self,scene):
        # ****************************************************************************************************
        self.scene = scene
        self.frame = 1
        self.dir_robot = self.dir+str(self.scene)+'/Robot_state/'
        self.dir_cam = self.dir+str(self.scene)+'/cam/cam_'
        self.dir_k_rgb = self.dir+str(self.scene)+'/kinect_rgb/Kinect_'
        self.dir_track = self.dir+str(self.scene)+'/tracking/'
        self.dir_cluster = self.dir+str(self.scene)+'/clusters/'

    def get_frame_str(self):
        # ****************************************************************************************************
        if self.frame<10:
            self.f_str = '000'+str(self.frame)
        elif self.frame<100:
            self.f_str = '00'+str(self.frame)
        elif self.frame<1000:
            self.f_str = '0'+str(self.frame)
        elif self.frame<10000:
            self.f_str = str(self.frame)

    def get_number_of_frames(self):
        # ****************************************************************************************************
        self.f_max = len([name for name in os.listdir(self.dir_robot) if os.path.isfile(os.path.join(self.dir_robot, name))])

    def get_number_of_objects(self):
        # ****************************************************************************************************
        self.obj_max = len([name for name in os.listdir(self.dir_cluster) if os.path.isfile(os.path.join(self.dir_cluster, name))])/4

    def read_single_frame(self,frame):
        # ****************************************************************************************************
        # initilise
        self.frame = frame
        self.get_frame_str()
        self.r_data[self.frame] = {}
        self.r_data[self.frame]['Left_Gripper'] = {}
        self.r_data[self.frame]['Right_Gripper'] = {}
        self.r_data[self.frame]['command'] = 0
        self.r_data[self.frame]['gripper'] = 100
        self.o_data[self.frame] = {}
        for i in range(self.obj_max):
            self.o_data[self.frame][i] = {}

        # reading the robot tracks data
        f = open(self.dir_robot+'Robot_state_'+self.f_str+'.txt', 'r')
        for i in range(50):
            data = f.readline().split('\n')[0]
            # reading the left gripper
            if i in range(21,29):
                self.r_data[self.frame]['Left_Gripper'][data.split(',')[0]]=float(data.split(',')[1])
            # reading the right gripper
            if i in range(31,39):
                self.r_data[self.frame]['Right_Gripper'][data.split(',')[0]]=float(data.split(',')[1])

            ##########################################################################################
            # FILTERING

            # if arm is not in control assume its static
            if frame != 1 and  i == 41:
                if data.split(',')[1] != 'right':
                    self.r_data[self.frame]['Right_Gripper'] = self.r_data[self.frame-1]['Right_Gripper']
                if data.split(',')[1] != 'left':
                    self.r_data[self.frame]['Left_Gripper'] = self.r_data[self.frame-1]['Left_Gripper']
                # getting gripper information
                if data.split(',')[1] == 'right':
                    if self.r_data[self.frame]['Right_Gripper']['R_gripper']>70:
                        self.r_data[self.frame]['gripper'] = 'open'
                    else:
                        self.r_data[self.frame]['gripper'] = 'close'
                elif data.split(',')[1] == 'left':
                    if self.r_data[self.frame]['Left_Gripper']['L_gripper']>70:
                        self.r_data[self.frame]['gripper'] = 'open'
                    else:
                        self.r_data[self.frame]['gripper'] = 'close'

            if i == 45:
                Command = float(data.split(',')[1].split('[')[1])

                # this means the arm shouldn't move as no command was given
                if frame != 1 and Command == 0:
                    self.r_data[self.frame] = self.r_data[self.frame-1]

                self.r_data[self.frame]['command'] = Command
                # print frame,Command,Command==0


        # reading the object tracks data
        for obj in range(self.obj_max):
            f = open(self.dir_track+'obj'+str(obj)+'_'+self.f_str+'.txt', 'r')
            if frame != 1 and self.r_data[frame]['command']==0:
                self.o_data[self.frame][obj] = self.o_data[self.frame-1][obj]
            elif frame != 1 and self.r_data[self.frame]['gripper']=='open':
                self.o_data[self.frame][obj] = self.o_data[self.frame-1][obj]
            else:
                for i in range(3):
                    data = f.readline().split('\n')[0]
                    self.o_data[self.frame][obj][data.split(':')[0]]=float(data.split(':')[1])


    def read_all_frames(self):
        # ****************************************************************************************************
        # initilise
        self.o_data = {}            # object tracks
        self.r_data = {}            # robot tracks
        self.get_number_of_frames()
        self.get_number_of_objects()

        # read data
        for frame in range(1,self.f_max+1):
            self.read_single_frame(frame)

        # for frame in range(1,self.f_max+1):
        #     print frame,self.r_data[frame]['command'],self.r_data[frame]['command']==0
        # print toto


    def apply_filter(self, window_length=3):
        """Once obtained the joint x,y,z coords.
        Apply a median filter over a temporal window to smooth the joint positions.
        Whilst doing this, create a world Trace object"""
        ################################################
        #right gripper
        X,Y,Z = [],[],[]
        for frame in self.r_data:
            X.append(self.r_data[frame]['Right_Gripper']['R_x'])
            Y.append(self.r_data[frame]['Right_Gripper']['R_y'])
            Z.append(self.r_data[frame]['Right_Gripper']['R_z'])

        t = np.linspace(0,1,len(X)) # create a time signal
        x1 = sp.signal.medfilt(X,21) # filter the signal
        y1 = sp.signal.medfilt(Y,21) # add noise to the signal
        z1 = sp.signal.medfilt(Z,21) # add noise to the signal

        for frame in self.r_data:
            self.r_data[frame]['Right_Gripper']['R_x'] = x1[frame-1]
            self.r_data[frame]['Right_Gripper']['R_y'] = y1[frame-1]
            self.r_data[frame]['Right_Gripper']['R_z'] = z1[frame-1]

        # plot the results
        plt.subplot(3,1,1)
        plt.plot(t,X,'yo-')
        plt.title('X')
        plt.xlabel('time')
        plt.subplot(3,1,1)
        plt.plot(t,x1,'bo-')
        plt.xlabel('time')

        # plot the results
        plt.subplot(3,1,2)
        plt.plot(t,Y,'yo-')
        plt.title('Y')
        plt.xlabel('time')
        plt.subplot(3,1,2)
        plt.plot(t,y1,'bo-')
        plt.xlabel('time')

        # plot the results
        plt.subplot(3,1,3)
        plt.plot(t,Z,'yo-')
        plt.title('Z')
        plt.xlabel('time')
        plt.subplot(3,1,3)
        plt.plot(t,z1,'bo-')
        plt.xlabel('time')

        plt.show()

        ################################################
        #left gripper
        X,Y,Z = [],[],[]
        for frame in self.r_data:
            X.append(self.r_data[frame]['Left_Gripper']['L_x'])
            Y.append(self.r_data[frame]['Left_Gripper']['L_y'])
            Z.append(self.r_data[frame]['Left_Gripper']['L_z'])

        x1 = sp.signal.medfilt(X,21) # add noise to the signal
        y1 = sp.signal.medfilt(Y,21) # add noise to the signal
        z1 = sp.signal.medfilt(Z,21) # add noise to the signal

        for frame in self.r_data:
            self.r_data[frame]['Left_Gripper']['L_x'] = x1[frame-1]
            self.r_data[frame]['Left_Gripper']['L_y'] = y1[frame-1]
            self.r_data[frame]['Left_Gripper']['L_z'] = z1[frame-1]

        ################################################
        # objects
        X,Y,Z = {},{},{}
        for frame in self.o_data:
            for obj in self.o_data[frame]:
                if obj not in X:
                    X[obj],Y[obj],Z[obj] = [],[],[]
                X[obj].append(self.o_data[frame][obj]['X'])
                Y[obj].append(self.o_data[frame][obj]['Y'])
                Z[obj].append(self.o_data[frame][obj]['Z'])

        x1,y1,z1 = {},{},{}
        for obj in self.o_data[frame]:
            x1[obj] = sp.signal.medfilt(X[obj],21) # add noise to the signal
            y1[obj] = sp.signal.medfilt(Y[obj],21) # add noise to the signal
            z1[obj] = sp.signal.medfilt(Z[obj],21) # add noise to the signal

        for frame in self.r_data:
            for obj in self.o_data[frame]:
                self.o_data[frame][obj]['X'] = x1[obj][frame-1]
                self.o_data[frame][obj]['Y'] = y1[obj][frame-1]
                self.o_data[frame][obj]['Z'] = z1[obj][frame-1]
