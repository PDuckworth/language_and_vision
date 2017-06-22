import cv2
import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorsys
from sklearn import mixture,metrics
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy.random import multivariate_normal


class locations():
    """docstring for locations."""
    def __init__(self):
        self.dir = "/home/omari/Datasets/Baxter_Dataset_final/scene"
        self.video = 203

    def _read_locations(self):
        dir1 = self.dir+str(self.video)+"/tracking/"
        files = sorted(glob.glob(dir1+"*.txt"))
        self.xyz = {}
        for f1 in files:
            f = open(f1,"r")
            for line in f:
                line = line.split("\n")[0]
                a,val = line.split(":")
                if a not in self.xyz:
                    self.xyz[a] = []
                self.xyz[a].append(float(val))
        print self.xyz

    def _cluster_locations(self):
        final_clf = 0
        best_v = 0
        self.X = []
        for x,y,z in zip(self.xyz["X"],self.xyz["Y"],self.xyz["Z"]):
            if self.X == []:
                self.X = [x,y,z]
            else:
                self.X = np.vstack((self.X,[x,y,z]))
        n_components_range = range(1, 10)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        lowest_bic = np.infty
        for cv_type in cv_types:
            for n_components in n_components_range:
                gmm = mixture.GaussianMixture(n_components=n_components,covariance_type=cv_type)
                gmm.fit(self.X)
                bic = gmm.bic(self.X)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
        clf = best_gmm
        Y_ = clf.predict(self.X)
        # pickle.dump( [clf,self.X], open( self.dir2+'colour_clusters.p', "wb" ) )
        self.final_clf = clf
        self.Y_ = list(Y_)
        for i in range(len(self.final_clf.means_)):
            print i,self.Y_.count(i)

    def _plot_locations(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = self.xyz["X"]
        ys = self.xyz["Y"]
        zs = self.xyz["Z"]

        c = []
        for i in range(len(xs)/4):
            a = i/float(len(xs))
            c.append((1.0-a , 0, a))
        for i in range(4):
            a = len(xs)/4*i
            b = len(xs)/4*(i+1)
            print len(c),len(xs[a:b])
            ax.scatter(xs[a:b], ys[a:b], zs[a:b], c=c, marker="o")

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_zlim([-.2,-.1])
        ax.set_xlim([.4,1])
        ax.set_ylim([-.3,.3])

    def _plot_clusters(self):
        self._HSV_tuples = [(x*1.0/5, 1, .7) for x in range(len(self.final_clf.means_))]
        self._colors = map(lambda x: colorsys.hsv_to_rgb(*x), self._HSV_tuples)

        self.rgb = []
        for i in self.Y_:
            self.rgb.append(self._colors[i])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = []
        Y = []
        Z = []
        for x,y,z in self.X:
            X.append(x)
            Y.append(y)
            Z.append(z)
        ax.scatter(X,Y,Z, c=self.rgb, marker='o')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_zlim([-.2,-.1])
        ax.set_xlim([.4,1])
        ax.set_ylim([-.3,.3])


        # table
        # x = [.4, .4, 1, 1]
        # y = [.4, -.4, -.4, .4]
        # z = [-.2, -.2, -.2, -.2]

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')

        # ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

def main():
    L = locations()
    L._read_locations()
    L._cluster_locations()
    L._plot_locations()
    L._plot_clusters()
    plt.show()

if __name__ == "__main__":
    main()
