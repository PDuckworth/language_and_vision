import cv2
import numpy as np
import glob
from operator import add
import pickle
from sklearn import mixture
# import itertools
from sklearn.metrics.cluster import v_measure_score
from sklearn import metrics
from sklearn import svm
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
import numpy.ma as ma
from numpy.random import uniform, seed
from matplotlib import cm

class locations():
    """docstring for shapes."""
    def __init__(self):
        self.dir = "/home/omari/Datasets/Baxter_Dataset_final/scene"
        self.dir_save = "/home/omari/Datasets/Baxter_Dataset_final/features/locations/"
        self.dir_scale = "/home/omari/Datasets/scalibility/Baxter/"
        self.th = 10
        self.sp = 2
        self.X = []     # fpfh vales
        self.XY = []
        self.eX = []    # esf vales
        self.GT = []
        self.shapes = {}
        self.images = []
        self.im_len = 60
        self.location_per_video = {}

    def _extract_object_images(self):
        f_x, f_y = 1212.9-700, 1129.0-700
        c_x, c_y = 187.3-700, 439.6-700
        for video in range(1,205):
            # print 'processing video: ',video
            dir1 = self.dir+str(video)+"/features/shapes/"
            img = cv2.imread(self.dir+str(video)+"/kinect_rgb/Kinect_0001.png")
            height, width = img.shape[:2]
            unique_objects = sorted(glob.glob(dir1+"fpfh*.pcd"))
            for obj in range(len(unique_objects)):
                # print obj
                dir2 = self.dir+str(video)+"/clusters/cloud_cluster_"+str(obj)
                tracks = sorted(glob.glob(dir2+".pcd"))
                for f1 in tracks:
                    X = []
                    Y = []
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
                    a = y2-y1
                    b = x2-x1
                    xo = (x2+x1)/2.0
                    yo = (y2+y1)/2.0
                    A = np.max([a,b])
                    x1,x2 = int(xo-A/2), int(xo+A/2)
                    y1,y2 = int(yo-A/2), int(yo+A/2)
                    canvas = img.copy()
                    # for i in range(x1,x2+1):
                    #     A = np.abs((i-(x2+x1)/2.0)/((x1+x2)/2.0-x1))
                    #     im[y1:y2,i:i+1,:] = [int(A*255),0,0]
                        # print i
                    # canvas = np.add(np.multiply(im,0.5,casting="unsafe"),np.multiply(img,0.5,casting="unsafe"),casting="unsafe")
                    canvas[y1:y2,x1-10:x1,:] = [0,0,0]
                    canvas[y1:y2,x2:x2+10,:] = [0,0,0]
                    canvas[y1-10:y1,x1:x2,:] = [0,0,0]
                    canvas[y2:y2+10,x1:x2,:] = [0,0,0]
                    cv2.line(canvas,(x1,y1),(x2,y2),(0,0,0),3)
                    cv2.line(canvas,(x1,y2),(x2,y1),(0,0,0),3)
                    canvas = canvas[:-60,120:-250,:]
                    cv2.imwrite(self.dir+str(video)+"/clusters/location_obj_"+str(obj)+".png",canvas)
                    canvas = cv2.resize(canvas, (60,60), interpolation = cv2.INTER_AREA)
                    self.images.append(canvas)

    def _read_locations(self):
        min_X = 0.607891
        max_X = 0.844499
        min_Y = -0.261884
        max_Y =  0.22944
        img_all = np.zeros((200,200,3),dtype=np.uint8)+255
        for video in range(1,205):
            self.location_per_video[video] = []
            dir1 = self.dir+str(video)+"/tracking/"
            dir2 = self.dir+str(video)+"/ground_truth/"
            files = sorted(glob.glob(dir1+"obj*_0001.txt"))
            #print files
            types = sorted(glob.glob(dir2+"GT_obj*.txt"))
            ground = sorted(glob.glob(dir2+"GT_location_*.txt"))
            # print ground
            obj = 0
            for f1,f2,f3 in zip(files,types,ground):
                img = np.zeros((200,200,3),dtype=np.uint8)+255
                f = open(f1,"r")
                xyz = []
                for line in f:
                    line = line.split("\n")[0]
                    a,val = line.split(":")
                    xyz.append(float(val))
                xyz = xyz[:-1]

                # plot the location
                x = 200-int((xyz[0]-min_X)/(max_X-min_X)*180+10)
                y = 200-int((xyz[1]-min_Y)/(max_Y-min_Y)*180+10)
                # check type
                f = open(f2,"r")
                for line in f:
                    line = line.split('\n')[0]
                    if line == "cup":
                        x += 35
                # print x,y
                th = 12
                l = 2
                img[x-th:x+th, y-l:y+l, :] = [0,0,255]
                img[x-l:x+l, y-th:y+th, :] = [0,0,255]
                img_all[x-th:x+th, y-l:y+l, :] = [0,0,255]
                img_all[x-l:x+l, y-th:y+th, :] = [0,0,255]

                # ## save the ground_truth
                # if x<100:
                #     val1 = "top_"
                # else:
                #     val1 = "centre_"
                #
                # if y < 200/3:
                #     val2 = "left"
                # elif y<200/3*2:
                #     val2 = ""
                # else:
                #     val2 = "right"
                # # print val1+val2
                # f1 = open(self.dir+str(video)+"/ground_truth/GT_location_"+str(obj)+".txt","w")
                # f1.write(val1+val2)
                # f1.close()


                if self.XY == []:
                    self.XY = [x,y]
                else:
                    self.XY = np.vstack((self.XY,[x,y]))

                if self.location_per_video[video] == []:
                    self.location_per_video[video] = xyz
                    self.location_per_video[video] = np.vstack((self.location_per_video[video],xyz))
                else:
                    self.location_per_video[video] = np.vstack((self.location_per_video[video],xyz))

                if self.X == []:
                    self.X = xyz
                else:
                    self.X = np.vstack((self.X,xyz))
                # print self.X
                f.close()

                f = open(f3,"r")
                for line in f:
                    line = line.split('\n')[0]
                    # if line == "orange":
                    #     print video,line
                    self.GT.append(line)
                    if line not in self.shapes:
                        self.shapes[line] = [xyz]
                    else:
                        self.shapes[line].append(xyz)
                f.close()

                # self.images.append(img)
                cv2.imwrite(self.dir+str(video)+"/clusters/loc_"+str(obj)+".png",img)
                obj+=1
        cv2.imwrite(self.dir_save+"all_locations.png",img_all)

    def _read_shapes_images(self):
        for video in range(1,205):
            dir1 = self.dir+str(video)+"/clusters/"
            files = sorted(glob.glob(dir1+"obj*.png"))
            for f1 in files:
                img = cv2.imread(f1)
                self.images.append(cv2.resize(img, (self.im_len,self.im_len), interpolation = cv2.INTER_AREA))

    def _plot_fpfh_values(self):
        th = self.th+15
        sp = self.sp+3
        for obj in self.shapes:
            L = len(self.shapes[obj])
            # L = 1
            img = np.zeros((th*L + sp*(L-1)+2*sp,   th*len(self.shapes[obj][0])+sp*(len(self.shapes[obj][0])-1)+2*sp,   3),dtype=np.uint8)
            for c1,i in enumerate(self.shapes[obj]):
                # print np.max(i)/100.0*255
                # if c1 == 30:
                    for c2,val in enumerate(i):
                        # c1 = 0
                        img[sp+c1*(th+sp):sp+c1*sp+(c1+1)*th,   sp+c2*(th+sp):sp+(c2+1)*th+c2*sp,  :] = int(val/100.0*255)+5
                # break
            cv2.imshow("img",img)
            cv2.imwrite(self.dir_save+"raw_features_"+obj+".png",img)

        for T in range(50):
            V = []
            count = 0
            for a1,a2 in zip(self.Y_,self.X):
                if a1 == T:
                    count+=1
                    if V == []:
                        V = a2
                    else:
                        V = map(add, V, a2)
            if count != 0:
                V = [x / float(count) for x in V]
                img = np.zeros((th*6+sp*5,  th*6+sp*5,  3),dtype=np.uint8)
                for c,val in enumerate(V):
                    a = np.mod(c,6)
                    b = int(c/6)
                    val = int(val/100.0*255)
                    img[a*(th+sp):(a+1)*th+a*sp,   b*(th+sp):(b+1)*th+b*sp,   :] = val
                cv2.imshow("img",img)
                cv2.imwrite(self.dir_save+"feature_"+str(T)+".png",img)

    def _cluster_locations(self):
        final_clf = 0
        best_v = 0
        X = self.X
        for i in range(10):
            print '#####',i
            ## 18 components did well!! 0.45
            n_components_range = range(3, 10)
            cv_types = ['spherical', 'tied', 'diag', 'full']
            lowest_bic = np.infty
            for cv_type in cv_types:
                for n_components in n_components_range:
                    gmm = mixture.GaussianMixture(n_components=n_components,covariance_type=cv_type)
                    gmm.fit(X)
                    # bic = gmm.bic(X)
                    # if bic < lowest_bic:
                    #     lowest_bic = bic
                    #     best_gmm = gmm
                    Y_ = gmm.predict(X)
                    v_meas = v_measure_score(self.GT, Y_)
                    if v_meas > best_v:
                        best_v = v_meas
                        final_clf = gmm
                        print best_v
                        print "clusters: ",len(final_clf.means_)
            # clf = best_gmm
            # Y_ = clf.predict(X)
            # v_meas = v_measure_score(self.GT, Y_)
            # if v_meas > best_v:
            #     best_v = v_meas
            #     final_clf = clf
            #     print best_v
        # print best_gmm
        self.best_v = best_v
        Y_ = final_clf.predict(X)
        # self.predictions = {}
        # for obj in self.GT:
        #     self.predictions[obj] = np.zeros(len(final_clf.means_))
        #     for i in self.faces[obj]:
        #         self.predictions[obj][final_clf.predict([i])[0]]+=1
        #
        # self.cost_matrix = np.zeros((len(self.faces),len(final_clf.means_)))
        #
        # for count,obj in enumerate(self.predictions):
        #     self.predictions[obj]/=np.sum(self.predictions[obj])
        #     self.cost_matrix[count] = self.predictions[obj]
        #

        pickle.dump( [final_clf,self.best_v], open( self.dir_save+'locations_clusters.p', "wb" ) )

    def _read_clusters(self):
        self.final_clf,self.best_v = pickle.load(open( self.dir_save+'locations_clusters.p', "rb" ) )
        print "number of clusters",len(self.final_clf.means_)
        self.Y_ = self.final_clf.predict(self.X)

        ## get the clusters in each video
        self.Y_per_video = {}
        for i in self.location_per_video:
            Y_ = []
            if not self.location_per_video[i] == []:
                X = self.final_clf.predict(self.location_per_video[i])
                for x in X:
                    if x not in Y_:
                        Y_.append(x)
                print i,Y_
            self.Y_per_video[i] = Y_
        pickle.dump( [len(self.final_clf.means_),self.Y_per_video] , open( self.dir_save+'clusters_per_video.p', "wb" ) )

        unique_clusters = []
        video = []
        for i in self.Y_per_video:
            for j in self.Y_per_video[i]:
                if j not in unique_clusters:
                    unique_clusters.append(j)
            video.append(len(unique_clusters))
        pickle.dump( video , open( self.dir_scale+'locations_per_video.p', "wb" ) )
        print video

    def _plot_clusters(self):
        self.clusters = {}
        for x,val in zip(self.XY,self.Y_):
            if val not in self.clusters:
                self.clusters[val] = np.zeros((200,200,3),dtype=np.uint8)
            a,b = x
            for i in range(10):
                self.clusters[val][a-i:a+i,b-i:b+i,:]+=1
        self.avg_images = {}
        for c in self.clusters:
            plt.matshow(self.clusters[c][:,:,0])
            plt.axis("off")
            plt.savefig(self.dir_save+'avg_'+str(c)+".png")
            self.avg_images[c] = cv2.imread(self.dir_save+'avg_'+str(c)+".png")

    def _pretty_plot(self):

        # Data to plot.
        self._plot_clusters()

        self.cluster_images = {}
        for img,val in zip(self.images,self.Y_):
            if val not in self.cluster_images:
                self.cluster_images[val] = []
            img = cv2.resize(img, (60,60), interpolation = cv2.INTER_AREA)
            self.cluster_images[val].append(img)

        for val in self.cluster_images:
            #self.cluster_images[val] = sorted(self.cluster_images[val])
            if len(self.cluster_images[val])>12:
                selected = []
                count = 0
                for i in range(0,len(self.cluster_images[val]),len(self.cluster_images[val])/12):
                    if count < 12:
                        selected.append(self.cluster_images[val][i])
                        count+=1
                self.cluster_images[val] = selected

        rows = 0
        maxi = 0
        for p in self.cluster_images:
            if len(self.cluster_images[p])>maxi:
                maxi = len(self.cluster_images[p])

        maxi = int(np.sqrt(maxi))
        for p in self.cluster_images:
            count = 0
            image = np.zeros((self.im_len*maxi,self.im_len*maxi,3),dtype=np.uint8)+255
            for i in range(maxi):
                for j in range(maxi):
                    if count < len(self.cluster_images[p]):
                        image[i*self.im_len:(i+1)*self.im_len,j*self.im_len:(j+1)*self.im_len,:] = self.cluster_images[p][count]
                    image[i*self.im_len:i*self.im_len+1,  j*self.im_len:(j+1)*self.im_len,  :] = 0
                    image[(i+1)*self.im_len:(i+1)*self.im_len+1,  j*self.im_len:(j+1)*self.im_len,  :] = 0
                    image[i*self.im_len:(i+1)*self.im_len,  j*self.im_len:j*self.im_len+1,  :] = 0
                    image[i*self.im_len:(i+1)*self.im_len,  (j+1)*self.im_len:(j+1)*self.im_len+1,  :] = 0
                    count+=1
            # cv2.imshow('test',image)
            # cv2.waitKey(2000)
            cv2.imwrite(self.dir_save+"cluster_"+str(p)+'.png',image)

        # self.cluster_images = {}
        # for img,val in zip(self.images,self.Y_):
        #     if val not in self.cluster_images:
        #         self.cluster_images[val] = []
        #     self.cluster_images[val].append(img)

        image_cluster_total = np.zeros((self.im_len*5*7,self.im_len*5*5,3),dtype=np.uint8)+255
        paper_img = np.zeros((self.im_len*5*2,self.im_len*5*4,3),dtype=np.uint8)+255
        # print len(self.cluster_images)
        # print iii
        count3 = 0
        for count2,p in enumerate(self.cluster_images):
            image_avg = self.avg_images[p][80:545,90:555,:]#np.zeros((self.im_len,self.im_len,3),dtype=np.uint8)
            MAX_NUMBER_OF_IMAGES_SHOWN = 14
            maxi = np.min([len(self.cluster_images[p]),MAX_NUMBER_OF_IMAGES_SHOWN])
            image_cluster = np.zeros((self.im_len*5,self.im_len*5,3),dtype=np.uint8)+255
            # print maxi
            for count,img in enumerate(self.cluster_images[p]):
                img[0:2,:,:]=0
                img[-2:,:,:]=0
                img[:,0:2,:]=0
                img[:,-2:,:]=0
                ang = count/float(maxi)*2*np.pi
                xc = int(1.95*self.im_len*np.cos(ang))
                yc = int(1.95*self.im_len*np.sin(ang))
                # print xc,yc
                C = int(2.5*self.im_len)
                x1 = int(xc-self.im_len/2.0+2.5*self.im_len)
                x2 = x1+self.im_len
                y1 = int(yc-self.im_len/2.0+2.5*self.im_len)
                y2 = y1+self.im_len
                cv2.line(image_cluster,(int(y1+y2)/2,int(x1+x2)/2),(C,C),(20,20,20),2)
                # print x1,x2,y1,y2
                image_cluster[x1:x2,y1:y2,:] = img
            # image_avg = cv2.resize(image_avg, (int(self.im_len*1.4),int(self.im_len*1.4)), interpolation = cv2.INTER_AREA)
            x1 = int((2.5-.9)*self.im_len)
            x2 = int(x1+1.8*self.im_len)
            # image_avg = cv2.imread(self.dir_save+"feature_"+str(p)+".png")
            image_avg = cv2.resize(image_avg, (int(self.im_len*1.8),int(self.im_len*1.8)), interpolation = cv2.INTER_AREA)
            image_cluster[x1:x2,x1:x2,:] = image_avg
            if count2<35:
                i1x = np.mod(count2,7)*self.im_len*5
                i2x = (np.mod(count2,7)+1)*self.im_len*5
                i1y = int(count2/7)*self.im_len*5
                i2y = (int(count2/7)+1)*self.im_len*5
                image_cluster_total[i1x:i2x,i1y:i2y,:] = image_cluster
                cv2.imwrite(self.dir_save+'all_clusters.jpg',image_cluster_total)

            if p in [0,1,2,3]:
                i1x = np.mod(count3,4)*self.im_len*5
                i2x = (np.mod(count3,4)+1)*self.im_len*5
                count3+=1
                i1y = 0
                i2y = self.im_len*5
                paper_img[i1y:i2y,i1x:i2x,:] = image_cluster
                cv2.imwrite(self.dir_save+'shapes_clusters_ex.jpg',paper_img)

            # cv2.imwrite(self.dir_save+str(p)+'_cluster.jpg',image_cluster)
            cv2.imwrite(self.dir_save+'cluster_images/'+str(p)+'_cluster.jpg',image_cluster)

    def _SVM(self):
        maxi = 0
        mean = 0
        for i in range(50):
            clf = svm.SVC(kernel='linear')
            l1 = int(-.25*len(self.X))
            l2 = int(.75*len(self.X))
            clf.fit(self.XY[:l1], self.GT[:l1])
            A = clf.predict(self.XY[l2:])
            mean += metrics.v_measure_score(self.GT[l2:], A)
        mean/=50
        print("supervised V-measure: %0.2f" % mean)

    def _print_results(self):
        #print v_measure_score(self.GT, self.Y_)
        true_labels = self.GT
        pred_labels = self.Y_
        print "\n supervised number of clusters:", len(set(true_labels))
        print("V-measure: %0.2f" % metrics.v_measure_score(true_labels, pred_labels))
        print("Homogeneity: %0.2f" % metrics.homogeneity_score(true_labels, pred_labels))
        print("Completeness: %0.2f" % metrics.completeness_score(true_labels, pred_labels))
        print("Mutual Information: %.2f" % metrics.mutual_info_score(true_labels, pred_labels))
        print("Adjusted Mutual Information: %0.2f" % metrics.normalized_mutual_info_score(true_labels, pred_labels))

def main():
    L = locations()
    L._extract_object_images()
    # # S._read_shapes()
    L._read_locations()
    # # S._read_shapes_images()
    #L._cluster_locations()
    L._read_clusters()
    # # S._plot_fpfh_values()
    L._pretty_plot()
    L._SVM()
    L._print_results()

if __name__=="__main__":
    main()
