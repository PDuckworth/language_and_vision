import cv2
import numpy as np
import glob
from operator import add

class shapes():
    """docstring for shapes."""
    def __init__(self):
        self.dir = "/home/omari/Datasets/Baxter_Dataset_final/scene"
        self.th = 40
        self.sp = 2
        self.X = []

    def _extract_object_images(self):
        f_x, f_y = 365.456, 365.456
        c_x, c_y = 264.878, 150.878#205.395
        X = 0.779842
        Y = -0.246483
        Z = -0.157207

        for video in range(1,60):
            dir1 = self.dir+str(video)+"/features/shapes/"
            img = cv2.imread(self.dir+str(video)+"/kinect_rgb/Kinect_0001.png")
            height, width = img.shape[:2]
            img = cv2.resize(img,(int(.6*width), int(.6*height)), interpolation = cv2.INTER_CUBIC)
            unique_objects = sorted(glob.glob(dir1+"fpfh*.pcd"))
            for obj in range(len(unique_objects)):
                dir2 = self.dir+str(video)+"/tracking/obj"+str(obj)
                tracks = sorted(glob.glob(dir2+"_0001.pcd"))
                for f1 in tracks:
                    f = open(f1,"r")
                    for count,line in enumerate(f):
                        line = line.split("\n")[0]
                        if count == 6:      # get width
                            num = int(line.split(" ")[1])
                        if count > 10:
                            xyz = map(float,line.split(" "))
                            x,y,z = xyz[:-1]
                            x_2d = int((x/z)*f_x + c_x)
                            y_2d = int((y/z)*f_y + c_y)
                            img[y_2d,x_2d,:]=255
                            test=1
                            # z,y,x = X,Y,Z
                            # x_2d = int((x/z)*f_x + c_x)
                            # y_2d = int((y/z)*f_y + c_y)
                            # img[y_2d-10:y_2d+10,x_2d,:]=0
                            # img[y_2d,x_2d-10:x_2d+10,:]=0


            cv2.imshow("img_kinect",img)
            cv2.waitKey(1000)

    def _read_shapes(self):
        for video in range(1,20):
            dir1 = self.dir+str(video)+"/features/shapes/"
            files = sorted(glob.glob(dir1+"fpfh*.pcd"))
            for f1 in files:
                num=1
                fpfh = 0
                f = open(f1,"r")
                for count,line in enumerate(f):
                    line = line.split("\n")[0]
                    if count == 6:      # get width
                        num = int(line.split(" ")[1])
                    if count == 11:
                        fpfh = map(float,line.split(" "))
                    if count > 11:
                        a = map(float,line.split(" "))
                        fpfh = map(add, fpfh, a)
                fpfh = [x / num for x in fpfh]
                if self.X == []:
                    self.X = fpfh
                else:
                    self.X = np.vstack((self.X,fpfh))
        # print self.X

    def _plot_fpfh_values(self):
        L = len(self.X)
        th = self.th
        sp = self.sp
        img = np.zeros((th*L + sp*(L-1),th*33,3),dtype=np.uint8)
        for c1,i in enumerate(self.X):
            for c2,val in enumerate(i):
                img[c1*(th+sp):c1*sp+(c1+1)*th,c2*th:(c2+1)*th,:] = int(val/100.0*255)
                # break

        cv2.imshow("img",img)
        cv2.waitKey(1000)


def main():
    S = shapes()
    # S._read_shapes()
    S._extract_object_images()
    # S._plot_fpfh_values()

if __name__=="__main__":
    main()
