import numpy as np
import itertools

from scipy import linalg
import matplotlib as mpl
#
import getpass
import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import colorsys
from sklearn import mixture
import operator
import pickle

class clustering():
    """docstring for clustering"""
    def __init__(self):
        self.username = getpass.getuser()
        self.dir1 = '/home/'+self.username+'/Datasets/ECAI_dataset/features/vid'
        self.dir2 = '/home/'+self.username+'/Datasets/ECAI_dataset/clusters/colours/'
        self.dir_text = '/home/'+self.username+'/Datasets/ECAI_dataset/ECAI_annotations/vid'
        self.folder = 1
        self.main_word = 'red'
        self.list_of_bad_words = []#['hair','watch','zipper','shoes','wristwatch','head']

    def _read_colours(self):
        self.R = []
        self.G = []
        self.B = []
        self.R_avg = []
        self.G_avg = []
        self.B_avg = []
        self.H = []
        self.L = []
        self.S = []
        self.c = []
        self.c_avg = []
        self.X = []
        self.Y = []
        self.Z = []
        self.X_avg = []
        self.Y_avg = []
        self.Z_avg = []
        self.frames = []
        for f in range(1,494):
            if self.main_word not in self.words[f]:
                continue
            ok = 1
            for word in self.list_of_bad_words:
                if word in self.words[f]:
                    ok=0
            if not ok:
                continue
            self.frames.append(f)
            self.folder = f
            f = open(self.dir1+str(self.folder)+'/colours.txt','r')
            bt,gt,rt = [],[],[]
            bb,gb,rb = [],[],[]
            top = 1
            for line in f:
                if '-' not in line:
                    line = line.split('\n')[0].split(',')
                    line = map(int, line)
                    # self.B.append(line[0])
                    # self.G.append(line[1])
                    # self.R.append(line[2])
                    # self.c.append([self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0])
                    # hsv = colorsys.rgb_to_hsv(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
                    # self._hsv_to_xyz(hsv)
                    if top:
                        bt.append(line[0])
                        gt.append(line[1])
                        rt.append(line[2])
                    else:
                        bb.append(line[0])
                        gb.append(line[1])
                        rb.append(line[2])
                else:
                    top = 0

            for i in range(int(len(bt)/10.0)):
                self.B.append(sum(bt[i*10:(i+1)*10])/10.0)
                self.G.append(sum(gt[i*10:(i+1)*10])/10.0)
                self.R.append(sum(rt[i*10:(i+1)*10])/10.0)
                self.c.append([self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0])
                # hsv = colorsys.rgb_to_hsv(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
                # self._hsv_to_xyz(hsv)
                hls = colorsys.rgb_to_hls(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
                self._hls_to_xyz(hls)

                self.B.append(sum(bb[i*10:(i+1)*10])/10.0)
                self.G.append(sum(gb[i*10:(i+1)*10])/10.0)
                self.R.append(sum(rb[i*10:(i+1)*10])/10.0)
                self.c.append([self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0])
                # hsv = colorsys.rgb_to_hsv(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
                # self._hsv_to_xyz(hsv)
                hls = colorsys.rgb_to_hls(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
                self._hls_to_xyz(hls)

            self.B_avg.append(sum(bt) / float(len(bt)))
            self.G_avg.append(sum(gt) / float(len(gt)))
            self.R_avg.append(sum(rt) / float(len(rt)))
            self.c_avg.append([self.R_avg[-1]/255.0, self.G_avg[-1]/255.0, self.B_avg[-1]/255.0])
            hls = colorsys.rgb_to_hls(self.R_avg[-1]/255.0, self.G_avg[-1]/255.0, self.B_avg[-1]/255.0)
            self._hls_to_xyz_avg(hls)

            self.B_avg.append(sum(bb) / float(len(bb)))
            self.G_avg.append(sum(gb) / float(len(gb)))
            self.R_avg.append(sum(rb) / float(len(rb)))
            self.c_avg.append([self.R_avg[-1]/255.0, self.G_avg[-1]/255.0, self.B_avg[-1]/255.0])
            hls = colorsys.rgb_to_hls(self.R_avg[-1]/255.0, self.G_avg[-1]/255.0, self.B_avg[-1]/255.0)
            self._hls_to_xyz_avg(hls)
            # self.H.append(hls[0])
            # self.L.append(hls[1])
            # self.S.append(hls[2])

            # hsv = colorsys.rgb_to_hsv(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
            # self._hsv_to_xyz(hsv)

            # self.B.append(sum(bb) / float(len(bb)))
            # self.G.append(sum(gb) / float(len(gb)))
            # self.R.append(sum(rb) / float(len(rb)))
            # self.c.append([self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0])
            #
            # hsv = colorsys.rgb_to_hsv(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
            # self._hsv_to_xyz(hsv)
            # hls = colorsys.rgb_to_hls(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
            # self._hls_to_xyz(hls)
            # self.H.append(hls[0])
            # self.L.append(hls[1])
            # self.S.append(hls[2])

    def _populate_red(self):
        X = np.random.rand(100,3)
        X[:,0]+=.5
        X[:,0]/=max(X[:,0])
        X[:,1]*=.1
        X[:,2]*=.1
        for i in X:
            hsv = colorsys.rgb_to_hsv(i[0], i[1], i[2])
            self.c.append([i[0], i[1], i[2]])
            self._hsv_to_xyz(hsv)

    def _populate_yellow(self):
        X = np.random.rand(100,3)
        X[:,0]*=.1
        X[:,0]+=.08
        X[:,1]*=.2
        X[:,1]+=.8
        X[:,2]*=.5
        X[:,2]+=.2
        for i in X:
            rgb = colorsys.hsv_to_rgb(i[0], i[1], i[2])
            self.c.append([rgb[0], rgb[1], rgb[2]])
            self._hsv_to_xyz(i)

    def _populate_green(self):
        X = np.random.rand(100,3)
        X*=30
        X-=15
        X[:,0]+=20
        X[:,1]+=50
        X[:,2]+=20

        X[:,0]/=255
        X[:,1]/=255
        X[:,2]/=255
        for i in X:
            hsv = colorsys.rgb_to_hsv(i[0], i[1], i[2])
            self.c.append([i[0], i[1], i[2]])
            self._hsv_to_xyz(hsv)

    def _populate_blue(self):
        X = np.random.rand(100,3)
        X[:,0]*=.2
        X[:,0]+=.5
        X[:,1]*=.2
        X[:,1]+=.4
        X[:,2]*=.5
        X[:,2]+=.2
        for i in X:
            rgb = colorsys.hsv_to_rgb(i[0], i[1], i[2])
            self.c.append([rgb[0], rgb[1], rgb[2]])
            self._hsv_to_xyz(i)

    def _populate_white(self):
        X = np.random.rand(100,3)
        X[:,0]*=.2
        X[:,0]+=.5
        X[:,1]*=.2
        X[:,1]+=.0
        X[:,2]*=.5
        X[:,2]+=.4
        for i in X:
            rgb = colorsys.hsv_to_rgb(i[0], i[1], i[2])
            self.c.append([rgb[0], rgb[1], rgb[2]])
            self._hsv_to_xyz(i)

    def _hsv_to_xyz(self,hsv):
        h = hsv[0]*2*np.pi
        s = hsv[1]
        v = hsv[2]
        z = v
        x = s*np.cos(h)
        y = s*np.sin(h)
        self.X.append(x)
        self.Y.append(y)
        self.Z.append(z)

    def _hls_to_xyz(self,hls):
        h = hls[0]*2*np.pi
        l = hls[1]
        s = hls[2]
        z = l
        x = s*np.cos(h)
        y = s*np.sin(h)
        self.X.append(x)
        self.Y.append(y)
        self.Z.append(z)

    def _hls_to_xyz_avg(self,hls):
        h = hls[0]*2*np.pi
        l = hls[1]
        s = hls[2]
        z = l
        x = s*np.cos(h)
        y = s*np.sin(h)
        self.X_avg.append(x)
        self.Y_avg.append(y)
        self.Z_avg.append(z)

    def _3d_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X, self.Y, self.Z, c=self.c, marker='o')

        ax.set_xlabel('B Label')
        ax.set_ylabel('G Label')
        ax.set_zlabel('R Label')

        plt.show()

    def _test(self):
        X = []
        for x,y,z in zip(self.X,self.Y,self.Z):
            if X == []:
                X.append([x,y,z])
            else:
                X = np.vstack((X,[x,y,z]))

        lowest_bic = np.infty
        bic = []
        n_components_range = range(2, 5)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type)
                gmm.fit(X)
                bic.append(gmm.bic(X))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

        bic = np.array(bic)

        self._HSV_tuples = [(x*1.0/11, 1.0, .7) for x in range(10)]
        self._colors = map(lambda x: colorsys.hsv_to_rgb(*x), self._HSV_tuples)

        clf = best_gmm
        bars = []

        # Plot the BIC scores
        # plt.figure()
        # spl = plt.subplot(111)
        # for i, (cv_type, color) in enumerate(zip(cv_types, self._colors)):
        #     xpos = np.array(n_components_range) + .2 * (i - 2)
        #     bars.append(plt.bar(xpos, bic[i * len(n_components_range):
        #                                   (i + 1) * len(n_components_range)],
        #                         width=.2, color=color))
        # plt.xticks(n_components_range)
        # plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        # plt.title('BIC score per model')
        # xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        #     .2 * np.floor(bic.argmin() / len(n_components_range))
        # spl.set_xlabel('Number of components')
        # spl.legend([b[0] for b in bars], cv_types)

        # Plot the winner
        # plt.figure()
        # ax = plt.subplot(111, projection='3d')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.scatter(self.X, self.Y, self.Z, c=self.c, marker='o')
        #
        # plt.figure()
        # ax = plt.subplot(111, projection='3d')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        X = []
        for x,y,z in zip(self.X_avg,self.Y_avg,self.Z_avg):
            if X == []:
                X.append([x,y,z])
            else:
                X = np.vstack((X,[x,y,z]))
        Y_ = clf.predict(X)
        f = open(self.dir2+self.main_word+"_colours.txt", "w")
        for i in range(len(Y_)/2):
            f.write(str(self.frames[i])+':'+str(Y_[2*i])+','+str(Y_[2*i+1])+'\n')
        f.close()

        maxi = {}
        for i in range(len(Y_)/2):
            a = Y_[2*i]
            b = Y_[2*i+1]
            if a not in maxi:
                maxi[a] = 0
            maxi[a] += 1
            if a!=b:
                if b not in maxi:
                    maxi[b] = 0
                maxi[b] += 1

        # for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
        #                                            self._colors)):
        #     if not np.any(Y_ == i):
        #         continue
        #     ax.scatter(X[Y_ == i, 0], X[Y_ == i, 1], X[Y_ == i, 2], c=color, marker='o')
        #     # print 'cluster:'+str(i)+',mean:',mean[0],mean[1],mean[2]

        f = open(self.dir2+self.main_word+"_colours_stats.txt", "w")
        sorted_x = sorted(maxi.items(), key=operator.itemgetter(1))
        for i in reversed(sorted_x):
            f.write(str(i[0])+','+str(float(i[1])/(len(Y_)/2))+'\n')
            print str(i[0])+','+str(float(i[1])/(len(Y_)/2))
        f.close()


        pickle.dump( [self.X,self.Y,self.Z,self.c,self.R,self.G,self.B,clf,Y_], open( self.dir2+self.main_word+"_colours.p", "wb" ) )

        # plt.title('Selected GMM: full model, '+str(clf.n_components)+' components')
        # plt.subplots_adjust(hspace=.35, bottom=.02)
        # plt.show()


    def _cluster(self):
        CLF = []
        MAXI_value = 0
        MAXI = {}
        Y_final = []
        X = []
        for x,y,z in zip(self.X,self.Y,self.Z):
            if X == []:
                X.append([x,y,z])
            else:
                X = np.vstack((X,[x,y,z]))

        X2 = []
        for x,y,z in zip(self.X_avg,self.Y_avg,self.Z_avg):
            if X2 == []:
                X2.append([x,y,z])
            else:
                X2 = np.vstack((X2,[x,y,z]))

        n_components_range = range(3, 5)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        if len(X)>15:
            for run in range(1):
                lowest_bic = np.infty
                bic = []
                for cv_type in cv_types:
                    for n_components in n_components_range:
                        # Fit a Gaussian mixture with EM
                        gmm = mixture.GaussianMixture(n_components=n_components,
                                                      covariance_type=cv_type)
                        gmm.fit(X)
                        bic.append(gmm.bic(X))
                        if bic[-1] < lowest_bic:
                            lowest_bic = bic[-1]
                            best_gmm = gmm
                bic = np.array(bic)
                clf = best_gmm
                bars = []
                Y_ = clf.predict(X2)
                maxi = {}
                for i in range(len(Y_)/2):
                    a = Y_[2*i]
                    b = Y_[2*i+1]
                    if a not in maxi:
                        maxi[a] = 0
                    maxi[a] += 1
                    if a!=b:
                        if b not in maxi:
                            maxi[b] = 0
                        maxi[b] += 1
                val = max(maxi.iteritems(), key=operator.itemgetter(1))[1]
                if val > MAXI_value:
                    MAXI_value = val
                    MAXI = maxi.copy()
                    Y_final = Y_
                    CLF = clf
        if self.main_word in ['yellow','beige','pink','greyish']:
            print self.main_word,len(Y_final)/2
            print Y_
        if len(Y_final)/2>10:

            self.all_passed_words.append(self.main_word)
            f = open(self.dir2+self.main_word+"_colours.txt", "w")
            for i in range(len(Y_final)/2):
                f.write(str(self.frames[i])+':'+str(Y_final[2*i])+','+str(Y_final[2*i+1])+'\n')
            f.close()

            f = open(self.dir2+self.main_word+"_colours_stats.txt", "w")
            sorted_x = sorted(MAXI.items(), key=operator.itemgetter(1))
            for i in reversed(sorted_x):
                f.write(str(i[0])+','+str(float(i[1])/(len(Y_final)/2))+'\n')
                # print str(i[0])+','+str(float(i[1])/(len(Y_final)/2))
            f.close()

            f = open(self.dir2+"0_words_for_colours.txt", "w")
            for i in self.all_passed_words:
                f.write(i+'\n')
            f.close()
            # print clf.means_
        else:
            pass
            # print self.main_word,'is studpid'

        pickle.dump( [self.X,self.Y,self.Z,self.c,self.R,self.G,self.B,CLF,Y_final], open( self.dir2+self.main_word+"_colours.p", "wb" ) )


    def _read_annotations(self):
        self.words = {}
        self.all_words = []
        self.all_passed_words = []
        for i in range(1,494):
            self.words[i] = []
            f = open(self.dir_text+str(i)+'/person.txt','r')
            for count,line in enumerate(f):
                if count == 0 or "(X)" in line or line == "\n":
                    continue
                line = line.split('\n')[0]
                if '#' in line:
                    print line
                    sys.exit(1)
                line = line.lower()
                line = line.replace('.','')
                line = line.replace(',','')
                line = line.replace('/','-')


                if 'hair' in line:
                    print i,line

                for word in line.split(' '):
                    if word != '' and word not in self.words[i]:
                        self.words[i].append(word)
                    if word != '' and word not in self.all_words:
                        self.all_words.append(word)

def main():
    f = clustering()
    f._read_annotations()
    for word in f.all_words:
        f.main_word = word
        f._read_colours()
    # f._populate_red()
    # f._populate_green()
    # f._populate_yellow()
    # f._populate_blue()
    # f._populate_white()
    # f._3d_plot()
    # f._test()
        # print word
        f._cluster()
    # f.create_sk_images()

if __name__=="__main__":
    main()
