# import cv2
# import numpy as np
#
#
# # mouse callback function
# def draw_circle(event,x,y,flags,param):
#     print 'test'
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(self.img_view,(x,y),100,(255,0,0),-1)
#

#

# cv2.namedWindow('image')
# # cv2.imshow('image',self.img_view)
# cv2.setMouseCallback('image ',draw_circle)
# while(1):
#     cv2.imshow('image',self.img_view)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()


import cv2
import numpy as np


class annotation():
    """docstring for annotation."""
    def __init__(self):
        self.drawing = False # true if mouse is pressed
        self.mode = True # if True, draw rectangle. Press 'm' to toggle to curve
        self.ix,self.iy = -1,-1
        self._read_data()
        self._control()
        self.to_save = []
        self.x0,self.x1,self.y0,self.y1 = 0,1000,0,500
        self._annotate()
        self.dir1 = "/home/omari/Downloads/test.csv"
        self.dir2 = "/home/omari/data.csv"

    def _read_data(self):
        f = open(self.dir1,"r")
        self.max_col = 0
        smallest_int = 10000
        self.data = {}
        self.labels = []
        for c,line in enumerate(f):
            line = line.split('\n')[0].split(",")
            if int(line[2]) > self.max_col:
                self.max_col = int(line[2])
            if int(line[2])-int(line[1]) < smallest_int:
                smallest_int = int(line[2])-int(line[1])
            self.data[line[0]+"_"+str(c)] = [int(line[1]),int(line[2])]
            self.labels.append(line[0]+"_"+str(c))

        self.img_data = np.zeros((len(self.data)*20,self.max_col,3),dtype=np.uint8)+255
        self.img_names = np.zeros((len(self.data)*20,200,3),dtype=np.uint8)+240
        font = cv2.FONT_HERSHEY_SIMPLEX
        for c,l in enumerate(self.labels):
            # l = l.split()
            self.img_data[c*20:(c+1)*20,self.data[l][0]:self.data[l][1],:] = (255,0,0)
            cv2.putText(self.img_names,l.split('_')[0],(10,c*20+12), font, .45,(0,0,0),1)
            self.img_names[c*20:c*20+1,:,:] = (0,0,0)
            self.img_data[c*20:c*20+1,:,:] = (0,0,0)


        # self.img_data[330:350,360:700,:] = 0
        self.img = np.concatenate((self.img_names[0:500,:,:], self.img_data[0:500,0:1000,:]), axis=1)
        # self.img = np.concatenate((self.img, self.img_control), axis=0)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.draw_circle)

    def _control(self):
        self.img_control = np.zeros((400,400,3),dtype=np.uint8)+250
        self.up = [(150,50),(250,100)]
        self.lf = [(50,120),(150,170)]
        self.dw = [(150,190),(250,240)]
        self.ri = [(250,120),(350,170)]
        self.nx = [(100,280),(300,330)]
        cv2.rectangle(self.img_control,self.up[0],self.up[1],(0,255,0),-1)
        cv2.rectangle(self.img_control,self.dw[0],self.dw[1],(255,0,0),-1)
        cv2.rectangle(self.img_control,self.ri[0],self.ri[1],(0,0,255),-1)
        cv2.rectangle(self.img_control,self.lf[0],self.lf[1],(255,255,0),-1)
        cv2.rectangle(self.img_control,self.nx[0],self.nx[1],(255,0,255),-1)
        cv2.namedWindow('control')
        cv2.setMouseCallback('control',self.draw_circle2)

    # mouse callback function
    def draw_circle(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            c = int(y/20)
            l = self.labels[c]
            print self.to_save
            if str(c) not in self.to_save:
                print 'you clicked on:',l.split('_')[0]
                self.img_data[c*20:(c+1)*20,self.data[l][0]:self.data[l][1],:] = (0,0,255)
                self.img_data[c*20:c*20+1,:,:] = (0,0,0)
                self.to_save.append(str(c))
            else:
                self.to_save.remove(str(c))
                self.img_data[c*20:(c+1)*20,self.data[l][0]:self.data[l][1],:] = (255,0,0)
                self.img_data[c*20:c*20+1,:,:] = (0,0,0)
            self.img = np.concatenate((self.img_names[self.y0:self.y1,:,:], self.img_data[self.y0:self.y1,self.x0:self.x1,:]), axis=1)

    def inside(self,val,range1):
        if val[0] > range1[0][0] and val[0] < range1[1][0] and val[1] > range1[0][1] and val[1] < range1[1][1]:
            return 1
        else:
            return 0

    # mouse callback function
    def draw_circle2(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.inside([x,y],self.up):
                print "up"
                if self.y0 > 100:
                    self.y1-=100
                    self.y0-=100
                elif self.y0 > 0:
                    self.y1=500
                    self.y0=0
            if self.inside([x,y],self.dw):
                print "down"
                if self.y1 < len(self.data)*20-100:
                    self.y1+=100
                    self.y0+=100
                elif self.y1 < len(self.data)*20:
                    self.y1=len(self.data)*20
                    self.y0=len(self.data)*20-500
            if self.inside([x,y],self.ri):
                print "right"
                if self.x1 < self.max_col-200:
                    self.x1+=200
                    self.x0+=200
                elif self.x1 < self.max_col:
                    self.x1=self.max_col
                    self.x0=self.max_col-1000

            if self.inside([x,y],self.lf):
                print "left"
                if self.x0 > 200:
                    self.x1-=200
                    self.x0-=200
                elif self.x0 > 0:
                    self.x1=1000
                    self.x0=0

            if self.inside([x,y],self.nx):
                print "next"
                self.img_data = np.zeros((len(self.data)*20,self.max_col,3),dtype=np.uint8)+255
                self.img_names = np.zeros((len(self.data)*20,200,3),dtype=np.uint8)+240
                font = cv2.FONT_HERSHEY_SIMPLEX
                for c,l in enumerate(self.labels):
                    self.img_data[c*20:(c+1)*20,self.data[l][0]:self.data[l][1],:] = (255,0,0)
                    cv2.putText(self.img_names,l,(10,c*20+12), font, .45,(0,0,0),1)
                    self.img_names[c*20:c*20+1,:,:] = (0,0,0)
                    self.img_data[c*20:c*20+1,:,:] = (0,0,0)
                strr = (",").join(self.to_save)+"\n"
                self.file_to_save = open(self.dir2,"a")
                self.file_to_save.write(strr)
                self.file_to_save.close()
                self.to_save = []
            self.img = np.concatenate((self.img_names[self.y0:self.y1,:,:], self.img_data[self.y0:self.y1,self.x0:self.x1,:]), axis=1)

    def _annotate(self):
        while(1):
            cv2.imshow('image',self.img)
            cv2.imshow('control',self.img_control)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('m'):
                self.mode = not self.mode
            elif k == 27:
                break
        cv2.destroyAllWindows()

def main():
    T = annotation()

if __name__ == '__main__':
    main()
