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
import math
import matplotlib.mlab as mlab
import operator
from scipy.interpolate import griddata
import numpy.ma as ma
from numpy.random import uniform, seed
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class directions():
    """docstring for shapes."""
    def __init__(self):
        self.dir = "/home/omari/Datasets/Baxter_Dataset_final/scene"
        self.dir_save = "/home/omari/Datasets/Baxter_Dataset_final/features/directions/"
        self.dir_scale = "/home/omari/Datasets/scalibility/Baxter/"
        self.th = 10
        self.sp = 2
        self.X = []     # fpfh vales
        self.xs,self.ys,self.zs = [],[],[]
        self.XY = []
        self.eX = []    # esf vales
        self.GT = []
        self.shapes = {}
        self.images = []
        self.im_len = 60
        self.directions_per_video = {}

    def _extract_object_images(self):
        f_x, f_y = 1212.9-700, 1129.0-700
        c_x, c_y = 187.3-700, 439.6-700
        for video in range(1,205):
            dir1 = self.dir+str(video)+"/tracking/"
            files = sorted(glob.glob(dir1+"obj*_0001.txt"))
            if len(files)>1:
                img = cv2.imread(self.dir+str(video)+"/kinect_rgb/Kinect_0001.png")
                print 'processing video: ',video
                dir1 = self.dir+str(video)+"/features/shapes/"
                height, width = img.shape[:2]
                unique_objects = sorted(glob.glob(dir1+"fpfh*.pcd"))
                image_locations = {}
                for obj in range(len(unique_objects)):
                    print obj
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
                        image_locations[obj] = [x1,x2,y1,y2]
                for i in range(len(files)-1):
                    canvas1 = img.copy()
                    x1,x2,y1,y2 = image_locations[i]
                    canvas1[y1:y2,x1-10:x1,:] = [0,0,0]
                    canvas1[y1:y2,x2:x2+10,:] = [0,0,0]
                    canvas1[y1-10:y1,x1:x2,:] = [0,0,0]
                    canvas1[y2:y2+10,x1:x2,:] = [0,0,0]
                    cv2.line(canvas1,(x1,y1),(x2,y2),(0,0,0),3)
                    cv2.line(canvas1,(x1,y2),(x2,y1),(0,0,0),3)
                    for j in range(i+1,len(files)):
                        canvas2 = canvas1.copy()
                        x1,x2,y1,y2 = image_locations[j]
                        canvas2[y1:y2,x1-10:x1,:] = [0,0,0]
                        canvas2[y1:y2,x2:x2+10,:] = [0,0,0]
                        canvas2[y1-10:y1,x1:x2,:] = [0,0,0]
                        canvas2[y2:y2+10,x1:x2,:] = [0,0,0]
                        cv2.line(canvas2,(x1,y1),(x2,y2),(0,0,0),3)
                        cv2.line(canvas2,(x1,y2),(x2,y1),(0,0,0),3)
                        canvas2 = canvas2[:-60,120:-250,:]
                        cv2.imwrite(self.dir+str(video)+"/clusters/distance_"+str(i)+"_"+str(j)+".png",canvas2)
                        canvas2 = cv2.resize(canvas2, (60,60), interpolation = cv2.INTER_AREA)
                        self.images.append(canvas2)

    def _read_directions(self):
        # min_X = 0.607891
        # max_X = 0.844499
        # min_Y = -0.261884
        # max_Y =  0.22944
        # img_all = np.zeros((200,200,3),dtype=np.uint8)+255
        ## make distances
        for video in range(1,205):
            self.directions_per_video[video] = []
            dir1 = self.dir+str(video)+"/tracking/"
            dir2 = self.dir+str(video)+"/ground_truth/"
            files = sorted(glob.glob(dir1+"obj*_0001.txt"))
            types = sorted(glob.glob(dir2+"GT_obj*.txt"))
            if len(files)>1:
                for i in range(len(files)):
                    for j in range(len(files)):
                        if i != j:
                            self.images.append(cv2.imread(self.dir+str(video)+"/clusters/distance_"+str(i)+"_"+str(j)+".png"))
                            o1 = open(files[i],'r')
                            o2 = open(files[j],'r')
                            xyz1, xyz2 = [], []
                            for line1,line2 in zip(o1,o2):
                                line1 = line1.split("\n")[0]
                                line2 = line2.split("\n")[0]
                                a1,val1 = line1.split(":")
                                a2,val2 = line2.split(":")
                                xyz1.append(float(val1))
                                xyz2.append(float(val2))

                            f = open(types[i],"r")
                            for line in f:
                                line = line.split('\n')[0]
                                if line == "cup":
                                    xyz1[2] += .1

                            f = open(types[j],"r")
                            for line in f:
                                line = line.split('\n')[0]
                                if line == "cup":
                                    xyz2[2] += .1

                            A = [i1-j1 for i1,j1 in zip(xyz2,xyz1)]
                            B = np.sqrt( A[0]**2 + A[1]**2 + A[2]**2 )
                            A = A/B
                            A[1]*=-1

                            self.xs.append(A[0])
                            self.ys.append(A[1])
                            self.zs.append(A[2])

                            if self.X == []:
                                self.X = A
                            else:
                                self.X = np.vstack((self.X,A))


                            if self.directions_per_video[video] == []:
                                self.directions_per_video[video] = A
                                self.directions_per_video[video] = np.vstack((self.directions_per_video[video],A))
                            else:
                                self.directions_per_video[video] = np.vstack((self.directions_per_video[video],A))

                            # ## save the ground_truth
                            # max_index, max_value = max(enumerate(np.abs(A)), key=operator.itemgetter(1))
                            # if max_index == 0:
                            #     if A[0]>0:
                            #         val1 = "front"
                            #     else:
                            #         val1 = "back"
                            #
                            # if max_index == 1:
                            #     if A[1]>0:
                            #         val1 = "right"
                            #     else:
                            #         val1 = "left"
                            #
                            # if max_index == 2:
                            #     if A[2]>0:
                            #         val1 = "top"
                            #     else:
                            #         val1 = "bottom"
                            #
                            # f1 = open(self.dir+str(video)+"/ground_truth/GT_direction_"+str(i)+"_"+str(j)+".txt","w")
                            # f1.write(val1)
                            # f1.close()

                            ## read GT
                            f1 = open(self.dir+str(video)+"/ground_truth/GT_direction_"+str(i)+"_"+str(j)+".txt","r")
                            for line in f1:
                                self.GT.append(line)
                            f1.close()

                files2 = sorted(glob.glob(dir1+"obj0_*.txt"))
                a = len(files2)
                if a<10:
                    b = "000"+str(a)
                elif a<100:
                    b = "00"+str(a)
                else:
                    b = "0"+str(a)
                files = sorted(glob.glob(dir1+"obj*_"+b+".txt"))
                # print files
                # print video,len(files2)
                for i in range(len(files)):
                    for j in range(len(files)):
                        if i != j:
                            o1 = open(files[i],'r')
                            o2 = open(files[j],'r')
                            xyz1, xyz2 = [], []
                            for line1,line2 in zip(o1,o2):
                                line1 = line1.split("\n")[0]
                                line2 = line2.split("\n")[0]
                                a1,val1 = line1.split(":")
                                a2,val2 = line2.split(":")
                                xyz1.append(float(val1))
                                xyz2.append(float(val2))

                            f = open(types[i],"r")
                            for line in f:
                                line = line.split('\n')[0]
                                if line == "cup":
                                    xyz1[2] += .1

                            f = open(types[j],"r")
                            for line in f:
                                line = line.split('\n')[0]
                                if line == "cup":
                                    xyz2[2] += .1

                            A = [i1-j1 for i1,j1 in zip(xyz2,xyz1)]
                            B = np.sqrt( A[0]**2 + A[1]**2 + A[2]**2 )
                            A = A/B
                            A[1]*=-1
                            # print i,j,A

                        if self.directions_per_video[video] == []:
                            self.directions_per_video[video] = A
                            self.directions_per_video[video] = np.vstack((self.directions_per_video[video],A))
                        else:
                            self.directions_per_video[video] = np.vstack((self.directions_per_video[video],A))
                # print self.directions_per_video[video]


        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # n = 100
        # for c, m, zl, zh in [('r', 'o', -50, -25)]:#, ('b', '^', -30, -5)]:
        #     # xs = randrange(n, 23, 32)
        #     # ys = randrange(n, 0, 100)
        #     # zs = randrange(n, zl, zh)
        #     ax.scatter(self.xs, self.ys, self.zs, c=c, marker=m)
        #
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        #
        # plt.show()

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

    def _cluster_directions(self):
        final_clf = 0
        best_v = 0
        X = self.X
        for i in range(10):
            print '#####',i
            ## 18 components did well!! 0.45
            n_components_range = range(2, 8)
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
        self.best_v = best_v
        Y_ = final_clf.predict(X)

        pickle.dump( [final_clf,self.best_v], open( self.dir_save+'directions_clusters.p', "wb" ) )

    def _read_clusters(self):
        self.final_clf,self.best_v = pickle.load(open( self.dir_save+'directions_clusters.p', "rb" ) )
        print "number of clusters",len(self.final_clf.means_)
        self.Y_ = self.final_clf.predict(self.X)
        ## get the clusters in each video
        self.Y_per_video = {}
        for i in self.directions_per_video:
            Y_ = []
            if not self.directions_per_video[i] == []:
                X = self.final_clf.predict(self.directions_per_video[i])
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
        pickle.dump( video , open( self.dir_scale+'directions_per_video.p', "wb" ) )
        print video

    def _plot_clusters(self):
        self.avg_images = {}
        sigma = np.sqrt(self.final_clf.covariances_[0])
        for c,i in enumerate(self.final_clf.means_):
            mu = i[0]

            x = np.linspace(0, .5, 500)
            y = mlab.normpdf(x, mu, sigma)
            x = list(x)
            y = list(y)
            x.append(0.5); y.append(0)
            x.append(0); y.append(0)
            fig, ax = plt.subplots()

            ax.fill(x, y, zorder=10)
            ax.grid(True, zorder=5)
            plt.savefig(self.dir_save+'avg_'+str(c)+".png")
            self.avg_images[c] = cv2.imread(self.dir_save+'avg_'+str(c)+".png")

    def _pretty_plot(self):
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
            cv2.imwrite(self.dir_save+"cluster_"+str(p)+'.png',image)

        image_cluster_total = np.zeros((self.im_len*5*7,self.im_len*5*5,3),dtype=np.uint8)+255
        paper_img = np.zeros((self.im_len*5*2,self.im_len*5*4,3),dtype=np.uint8)+255
        count3 = 0
        for count2,p in enumerate(self.cluster_images):
            image_avg = self.avg_images[p][85:1000,90:725,:]#np.zeros((self.im_len,self.im_len,3),dtype=np.uint8)
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
            image_avg[0:2,:,:]=0
            image_avg[-2:,:,:]=0
            image_avg[:,0:2,:]=0
            image_avg[:,-2:,:]=0
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
                cv2.imwrite(self.dir_save+'distances_clusters_ex.jpg',paper_img)

            # cv2.imwrite(self.dir_save+str(p)+'_cluster.jpg',image_cluster)
            cv2.imwrite(self.dir_save+'cluster_images/'+str(p)+'_cluster.jpg',image_cluster)

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

    def _SVM(self):
        maxi = 0
        mean = 0
        for i in range(50):
            clf = svm.SVC(kernel='linear')
            l1 = int(-.25*len(self.X))
            l2 = int(.75*len(self.X))
            clf.fit(self.X[:l1], self.GT[:l1])
            A = clf.predict(self.X[l2:])
            mean += metrics.v_measure_score(self.GT[l2:], A)
        mean/=50
        print("supervised V-measure: %0.2f" % mean)

def main():
    D = directions()
    D._read_directions()
    # d._extract_object_images()
    # D._cluster_directions()
    D._read_clusters()
    # d._plot_clusters()
    # d._pretty_plot()
    # D._print_results()
    # D._SVM()

if __name__=="__main__":
    main()
