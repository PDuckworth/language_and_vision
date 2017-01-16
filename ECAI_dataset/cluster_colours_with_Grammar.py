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
        self.dir1 = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/features/vid'
        self.dir2 = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/clusters/colours/'
        self.dir_grammar = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/grammar/'
        self.dir_text = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/ECAI_annotations/vid'
        self.folder = 1
        self.main_word = 'red'

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
            ok_word = []
            if self.main_word in self.words_top[f]:
                ok_word.append('top')
            if self.main_word in self.words_low[f]:
                ok_word.append('low')
            if ok_word == []:
                continue
            self.words_count[self.main_word]+=1
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
            inter = 5
            if 'top' in ok_word:
                for i in range(int(len(bt)/float(inter))):
                    self.B.append(sum(bt[i*inter:(i+1)*inter])/float(inter))
                    self.G.append(sum(gt[i*inter:(i+1)*inter])/float(inter))
                    self.R.append(sum(rt[i*inter:(i+1)*inter])/float(inter))
                    self.c.append([self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0])
                    # hsv = colorsys.rgb_to_hsv(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
                    # self._hsv_to_xyz(hsv)
                    hls = colorsys.rgb_to_hls(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
                    self._hls_to_xyz(hls)
            if 'low' in ok_word:
                for i in range(int(len(bb)/float(inter))):
                    self.B.append(sum(bb[i*inter:(i+1)*inter])/float(inter))
                    self.G.append(sum(gb[i*inter:(i+1)*inter])/float(inter))
                    self.R.append(sum(rb[i*inter:(i+1)*inter])/float(inter))
                    self.c.append([self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0])
                    # hsv = colorsys.rgb_to_hsv(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
                    # self._hsv_to_xyz(hsv)
                    hls = colorsys.rgb_to_hls(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
                    self._hls_to_xyz(hls)

            if 'top' in ok_word:
                self.B_avg.append(sum(bt) / float(len(bt)))
                self.G_avg.append(sum(gt) / float(len(gt)))
                self.R_avg.append(sum(rt) / float(len(rt)))
                self.c_avg.append([self.R_avg[-1]/255.0, self.G_avg[-1]/255.0, self.B_avg[-1]/255.0])
                hls = colorsys.rgb_to_hls(self.R_avg[-1]/255.0, self.G_avg[-1]/255.0, self.B_avg[-1]/255.0)
                self._hls_to_xyz_avg(hls)

            if 'low' in ok_word:
                self.B_avg.append(sum(bb) / float(len(bb)))
                self.G_avg.append(sum(gb) / float(len(gb)))
                self.R_avg.append(sum(rb) / float(len(rb)))
                self.c_avg.append([self.R_avg[-1]/255.0, self.G_avg[-1]/255.0, self.B_avg[-1]/255.0])
                hls = colorsys.rgb_to_hls(self.R_avg[-1]/255.0, self.G_avg[-1]/255.0, self.B_avg[-1]/255.0)
                self._hls_to_xyz_avg(hls)


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

        n_components_range = range(2, 5)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        if len(X)>15:
            for run in range(1):
                lowest_bic = np.infty
                bic = []
                for cv_type in cv_types:
                    for n_components in n_components_range:
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
                for a in Y_:
                    if a not in maxi:
                        maxi[a] = 0
                    maxi[a] += 1
                val = max(maxi.iteritems(), key=operator.itemgetter(1))[1]
                if val > MAXI_value:
                    MAXI_value = val
                    MAXI = maxi.copy()
                    Y_final = Y_
                    CLF = clf
        print MAXI_value

        f = open(self.dir2+self.main_word+"_colours_stats.txt", "w")
        sorted_x = sorted(MAXI.items(), key=operator.itemgetter(1))
        for i in reversed(sorted_x):
            f.write(str(i[0])+','+str(float(i[1])/(len(Y_final)/2))+'\n')
        f.close()

        pickle.dump( [self.X,self.Y,self.Z,self.c,self.R,self.G,self.B,CLF,Y_final], open( self.dir2+self.main_word+"_colours.p", "wb" ) )

    def _3d_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X, self.Y, self.Z, c=self.c, marker='o')

        ax.set_xlabel('B Label')
        ax.set_ylabel('G Label')
        ax.set_zlabel('R Label')

        plt.show()

    def _read_annotations(self):
        self.all_words = []
        for i in range(1,494):
            f = open(self.dir_text+str(i)+'/person.txt','r')
            for count,line in enumerate(f):
                if count == 0 or "(X)" in line or line == "\n":
                    continue
                line = line.split('\n')[0]
                if '#' in line:
                    sys.exit(1)
                line = line.lower()
                line = line.replace('.','')
                line = line.replace(',','')
                line = line.replace('/','-')
                for word in line.split(' '):
                    if word != '' and word not in self.all_words:
                        self.all_words.append(word)

    def _read_tags(self):
        self.words_top = {}
        self.words_low = {}
        self.tags,self.words_count = pickle.load(open( self.dir_grammar+"tags.p", "rb" ) )
        for i in self.tags.keys():
            self.words_top[i] = []
            self.words_low[i] = []
            for word in self.tags[i]['upper_garment']:
                if word not in self.words_top[i]:
                    self.words_top[i].append(word)
            for word in self.tags[i]['lower_garment']:
                if word not in self.words_low[i]:
                    self.words_low[i].append(word)

def main():
    f = clustering()
    f._read_annotations()
    f._read_tags()
    f.words_count = {}
    for word in f.all_words:
        f.words_count[word] = 0
        f.main_word = word
        f._read_colours()
        if f.words_count[word]>10:
            print word,f.words_count[word]
            f._cluster()
            f._3d_plot()

if __name__=="__main__":
    main()
