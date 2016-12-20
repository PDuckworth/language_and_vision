import cv2
import numpy as np
import getpass
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



class features():
    """docstring for features"""
    def __init__(self):
        self.username = getpass.getuser()
        self.dir1 = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/'
        self.folder = 201
        self.shirt_mean = {}
        self.short_mean = {}
        self._read_video()

        self.shirt_joints = [
            'torso',
            'left_shoulder',
            'right_shoulder']

        self.short_joints = [
            'left_knee',
            'torso',
            'right_knee']

    def _read_video(self):
        self.skeleton_data = {}
        self.dir_img = self.dir1+'/vid'+str(self.folder)+'/images/'
        self.dir_skl = self.dir1+'/vid'+str(self.folder)+'/skeleton/'
        self.imgs = sorted([f for f in os.listdir(self.dir_img) if '.jpg' in f])
        self.skls = sorted([f for f in os.listdir(self.dir_skl) if '.txt' in f])
        self.shirt = []
        self.short = []
        self.shirt_mean[self.folder] = {}
        self.short_mean[self.folder] = {}

    def create_sk_images(self):
        for folder in [50,409]:
            self.folder = folder
            self._read_video()
            for frame in range(len(self.imgs)):
                print frame
                # if frame>40: break
                rgb_img = cv2.imread(self.dir_img+self.imgs[frame])
                self.get_2d_sk(frame)
                img_skl = self.get_shirt_short(rgb_img)
                self._get_mean_colors(frame)
                cv2.imshow('test',img_skl)
                cv2.waitKey(10)
        self._plot_rgb_data()

    def _plot_rgb_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        b = []
        g = []
        r = []
        c = []

        for folder in [50,409]:
            for i in self.shirt_mean[folder]:
                b.append(self.shirt_mean[folder][i][0])
                g.append(self.shirt_mean[folder][i][1])
                r.append(self.shirt_mean[folder][i][2])
                c.append([r[-1]/255.0, g[-1]/255.0, b[-1]/255.0])
                b.append(self.short_mean[folder][i][0])
                g.append(self.short_mean[folder][i][1])
                r.append(self.short_mean[folder][i][2])
                c.append([r[-1]/255.0, g[-1]/255.0, b[-1]/255.0])
        ax.scatter(b, g, r, c=c, marker='o')

        ax.set_xlabel('B Label')
        ax.set_ylabel('G Label')
        ax.set_zlabel('R Label')

        plt.show()


    def _get_mean_colors(self,frame):
        # finding shirt mean
        B = self.shirt[:,:,0]!=0
        G = self.shirt[:,:,1]!=0
        R = self.shirt[:,:,2]!=0
        B_mean = int(np.mean(self.shirt[R*B*G][:,0]))
        G_mean = int(np.mean(self.shirt[R*B*G][:,1]))
        R_mean = int(np.mean(self.shirt[R*B*G][:,2]))
        self.shirt_mean[self.folder][frame] = [B_mean,G_mean,R_mean]
        # finding short mean
        B = self.short[:,:,0]!=0
        G = self.short[:,:,1]!=0
        R = self.short[:,:,2]!=0
        B_mean = int(np.mean(self.short[R*B*G][:,0]))
        G_mean = int(np.mean(self.short[R*B*G][:,1]))
        R_mean = int(np.mean(self.short[R*B*G][:,2]))
        self.short_mean[self.folder][frame] = [B_mean,G_mean,R_mean]

    def get_shirt_short(self, img):
        img = self._remove_black(img,[1,1,1])
        mask_shirt = np.zeros((480,640), np.uint8)
        points = []
        for j in self.shirt_joints:
            y = self.skeleton_data[j][4]
            x = self.skeleton_data[j][3]
            points.append([x,y])
        poly = np.array(points, np.int32)
        cv2.fillConvexPoly(mask_shirt, poly, (255,255,255))
        shirt = cv2.bitwise_and(img, img, mask=mask_shirt)
        self.shirt = self._remove_black(shirt,[0,0,0])
        cv2.imshow("shirt", self.shirt)

        mask_short = np.zeros((480,640), np.uint8)
        points = []
        for j in self.short_joints:
            x = self.skeleton_data[j][3]
            y = self.skeleton_data[j][4]
            points.append([x,y])
        poly = np.array(points, np.int32)
        cv2.fillConvexPoly(mask_short, poly, (255,0,255))
        short = cv2.bitwise_and(img, img, mask=mask_short)
        self.short = self._remove_black(short,[0,0,0])
        cv2.imshow("short", self.short)
        return img

    def _remove_black(self,img,val):
        B = img[:,:,0]==0
        G = img[:,:,1]==0
        R = img[:,:,2]==0
        img[B*G*R] = val
        return img


    def get_2d_sk(self,val):
        fx = 525.0
        fy = 525.0
        cx = 319.5
        cy = 239.5
        f1 = open(self.dir_skl+self.skls[val],'r')
        for count,line in enumerate(f1):
            # read the joint name
            if (count-1)%10 == 0:
                j = line.split('\n')[0]
                self.skeleton_data[j] = [0,0,0,0,0]
            # read the x value
            elif (count-1)%10 == 2:
                a = float(line.split('\n')[0].split(':')[1])
                self.skeleton_data[j][0] = a
            # read the y value
            elif (count-1)%10 == 3:
                a = float(line.split('\n')[0].split(':')[1])
                self.skeleton_data[j][1] = a
            # read the z value
            elif (count-1)%10 == 4:
                a = float(line.split('\n')[0].split(':')[1])
                self.skeleton_data[j][2] = a
                #2D data
                x = self.skeleton_data[j][0]
                y = self.skeleton_data[j][1]
                z = self.skeleton_data[j][2]
                x2d = int(x*fx/z*1 +cx);
                y2d = int(y*fy/z*-1+cy);
                self.skeleton_data[j][3] = x2d
                self.skeleton_data[j][4] = y2d
def main():
    f = features()
    f.create_sk_images()

if __name__=="__main__":
    main()
