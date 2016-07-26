import numpy as np
import cv2
import getpass
import os, os.path

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
            # FITERING

            # if arm is not in control assume its static
            if frame != 1 and  i == 41:
                if data.split(',')[1] != 'right':
                    self.r_data[self.frame]['Right_Gripper'] = self.r_data[self.frame-1]['Right_Gripper']
                if data.split(',')[1] != 'left':
                    self.r_data[self.frame]['Left_Gripper'] = self.r_data[self.frame-1]['Left_Gripper']

            if i == 45:
                Command = float(data.split(',')[1].split('[')[1])
                print frame,Command

                # this means the arm shouldn't move as no command was given
                if frame != 1 and Command == 0:
                    self.r_data[self.frame] = self.r_data[self.frame-1]

        # reading the object tracks data
        for obj in range(self.obj_max):
            f = open(self.dir_track+'obj'+str(obj)+'_'+self.f_str+'.txt', 'r')
            for i in range(3):
                data = f.readline().split('\n')[0]
                self.o_data[self.frame][obj][data.split(':')[0]]=float(data.split(':')[1])
        #
        # if frame != 1:



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
