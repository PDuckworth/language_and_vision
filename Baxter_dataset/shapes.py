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
        f_x, f_y = 1212.9-700, 1129.0-700
        c_x, c_y = 187.3-700, 439.6-700

        # def nothing(x):
            # pass
        for video in range(1,205):
            print 'processing video: ',video
            dir1 = self.dir+str(video)+"/features/shapes/"
            img = cv2.imread(self.dir+str(video)+"/kinect_rgb/Kinect_0001.png")
            height, width = img.shape[:2]
            # img = cv2.resize(img,(int(.6*width), int(.6*height)), interpolation = cv2.INTER_CUBIC)
            unique_objects = sorted(glob.glob(dir1+"fpfh*.pcd"))
            for obj in range(len(unique_objects)):
                print obj
                dir2 = self.dir+str(video)+"/clusters/cloud_cluster_"+str(obj)
                tracks = sorted(glob.glob(dir2+".pcd"))
                for f1 in tracks:
                    X = []
                    Y = []
                    # Z = []
                    f = open(f1,"r")
                    for count,line in enumerate(f):
                        line = line.split("\n")[0]
                        if count == 6:      # get width
                            num = int(line.split(" ")[1])
                        if count > 10:
                            xyz = map(float,line.split(" "))
                            x,y,z = xyz[:-1]
                            # X.append(z)
                            # Y.append(y)
                            # Z.append(x)
                        # for z,y,x in zip(X,Y,Z):
                            x_2d = int((x/z)*f_x + c_x)
                            y_2d = int((y/z)*f_y + c_y)
                            if x_2d < 0:
                                x_2d += width
                            if y_2d < 0:
                                y_2d += height
                            X.append(x_2d)
                            Y.append(y_2d)
                    x1,x2 = np.min(X)-20,np.max(X)+20
                    y1,y2 = np.min(Y)-20,np.max(Y)+20
                    cv2.imwrite(self.dir+str(video)+"/clusters/obj_"+str(obj)+".png",img[y1:y2,x1:x2,:])
                    # print x1,y1, "--" , x2,y2
                    # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

                    # for x_2d,y_2d in zip(X,Y):
                    #     try:
                    #         img[y_2d,x_2d,:]=0
                    #         img[y_2d,x_2d,:]=0
                    #     except:
                    #         print 'bad val'

            # cv2.imshow('image',img)
            # k = cv2.waitKey(1000) & 0xFF
            # if k == 27:
            #     break

        # # Create a black image, a window
        # img = cv2.imread(self.dir+str(video)+"/kinect_rgb/Kinect_0001.png")
        # height, width = img.shape[:2]
        # img = cv2.resize(img,(int(.6*width), int(.6*height)), interpolation = cv2.INTER_CUBIC)
        # cv2.namedWindow('image')
        #
        # # create trackbars for color change
        # cv2.createTrackbar('fx','image',0,14000,nothing)
        # cv2.createTrackbar('fy','image',0,14000,nothing)
        # cv2.createTrackbar('cx','image',0,14000,nothing)
        # cv2.createTrackbar('cy','image',0,14000,nothing)
        #
        # while(1):
        #     img = cv2.imread(self.dir+str(video)+"/kinect_rgb/Kinect_0001.png")
        #
        #     # get current positions of four trackbars
        #     f_x = cv2.getTrackbarPos('fx','image')/10.0 - 700
        #     f_y = cv2.getTrackbarPos('fy','image')/10.0 - 700
        #     c_x = cv2.getTrackbarPos('cx','image')/10.0 - 700
        #     c_y = cv2.getTrackbarPos('cy','image')/10.0 - 700



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
