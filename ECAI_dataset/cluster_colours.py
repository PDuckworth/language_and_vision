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

import pickle

class clustering():
    """docstring for clustering"""
    def __init__(self):
        self.username = getpass.getuser()
        self.dir1 = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/features/vid'
        self.dir2 = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/clusters/colours/'
        self.folder = 1

    def _read_colours(self):
        self.R = []
        self.G = []
        self.B = []
        self.H = []
        self.L = []
        self.S = []
        self.c = []
        self.X = []
        self.Y = []
        self.Z = []
        for f in range(1,494):
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
            # print bt

            # print 'b top:',sum(bt) / float(len(bt))
            # print 'g top:',sum(gt) / float(len(gt))
            # print 'r top:',sum(rt) / float(len(rt))
            # print
            # print 'b bot:',sum(bb) / float(len(bb))
            # print 'g bot:',sum(gb) / float(len(gb))
            # print 'r bot:',sum(rb) / float(len(rb))


            self.B.append(sum(bt) / float(len(bt)))
            self.G.append(sum(gt) / float(len(gt)))
            self.R.append(sum(rt) / float(len(rt)))
            self.c.append([self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0])

            # hls = colorsys.rgb_to_hls(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
            # self._hls_to_xyz(hls)
            # self.H.append(hls[0])
            # self.L.append(hls[1])
            # self.S.append(hls[2])

            hsv = colorsys.rgb_to_hsv(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
            self._hsv_to_xyz(hsv)

            self.B.append(sum(bb) / float(len(bb)))
            self.G.append(sum(gb) / float(len(gb)))
            self.R.append(sum(rb) / float(len(rb)))
            self.c.append([self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0])

            hsv = colorsys.rgb_to_hsv(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
            self._hsv_to_xyz(hsv)
            # hls = colorsys.rgb_to_hls(self.R[-1]/255.0, self.G[-1]/255.0, self.B[-1]/255.0)
            # self._hls_to_xyz(hls)
            # self.H.append(hls[0])
            # self.L.append(hls[1])
            # self.S.append(hls[2])

    def _populate_red(self):
        X = np.random.rand(30,3)
        X[:,0]+=.5
        X[:,0]/=max(X[:,0])
        X[:,1]*=.1
        X[:,2]*=.1
        for i in X:
            hsv = colorsys.rgb_to_hsv(i[0], i[1], i[2])
            self.c.append([i[0], i[1], i[2]])
            self._hsv_to_xyz(hsv)

    def _populate_yellow(self):
        X = np.random.rand(50,3)
        X*=30
        X-=15
        X[:,0]+=115
        X[:,1]+=80
        X[:,2]+=20

        X[:,0]/=255
        X[:,1]/=255
        X[:,2]/=255
        for i in X:
            hsv = colorsys.rgb_to_hsv(i[0], i[1], i[2])
            self.c.append([i[0], i[1], i[2]])
            self._hsv_to_xyz(hsv)

    def _populate_green(self):
        X = np.random.rand(50,3)
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
        n_components_range = range(2, 7)
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

        self._HSV_tuples = [(x*1.0/11, 1.0, .7) for x in range(8)]
        self._colors = map(lambda x: colorsys.hsv_to_rgb(*x), self._HSV_tuples)

        # color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
        #                               'darkorange','red','green','black'])
        clf = best_gmm
        bars = []

        # Plot the BIC scores
        spl = plt.subplot(3, 1, 1)
        for i, (cv_type, color) in enumerate(zip(cv_types, self._colors)):
            xpos = np.array(n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)],
                                width=.2, color=color))
        plt.xticks(n_components_range)
        plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        plt.title('BIC score per model')
        xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
            .2 * np.floor(bic.argmin() / len(n_components_range))
        # plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], cv_types)
        # print clf

        # Plot the winner
        ax = plt.subplot(3, 1, 2, projection='3d')

        # fig = plt.figure()
        ax.scatter(self.X, self.Y, self.Z, c=self.c, marker='o')
        #

        ax = plt.subplot(3, 1, 3, projection='3d')
        # ax.set_xlabel('B Label')
        # ax.set_ylabel('G Label')
        # ax.set_zlabel('R Label')

        # plt.show()

        Y_ = clf.predict(X)
        for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                                   self._colors)):
            print i,color
            print X[Y_ == i, 0]
            if not np.any(Y_ == i):
                continue
            ax.scatter(X[Y_ == i, 0], X[Y_ == i, 1], X[Y_ == i, 2], c=color, marker='o')

            # Plot an ellipse to show the Gaussian component
            # angle = np.arctan2(w[0][1], w[0][0])
            # angle = 180. * angle / np.pi  # convert to degrees
            # v = 2. * np.sqrt(2.) * np.sqrt(v)
            # ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            # ell.set_clip_box(splot.bbox)
            # ell.set_alpha(.5)
            # splot.add_artist(ell)

        pickle.dump( [self.X,self.Y,self.Z,self.c,self.R,self.G,self.B,clf,Y_], open( self.dir2+"colours.p", "wb" ) )

        plt.xticks(())
        plt.yticks(())
        plt.title('Selected GMM: full model, '+str(clf.n_components)+' components')
        plt.subplots_adjust(hspace=.35, bottom=.02)
        plt.show()



def main():
    f = clustering()
    f._read_colours()
    f._populate_red()
    f._populate_green()
    f._populate_yellow()
    f._3d_plot()
    f._test()
    # f.create_sk_images()

if __name__=="__main__":
    main()
