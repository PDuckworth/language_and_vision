import numpy as np
from sklearn import mixture
import itertools
from sklearn.metrics.cluster import v_measure_score
import cv2
import pickle

class faces_class():
    """docstring for faces"""
    def __init__(self):
        self.dir_faces =  '/home/omari/Datasets_old/ECAI_dataset_segmented/faces/'
        self.dir_grammar = '/home/omari/Datasets_old/ECAI_dataset_segmented/grammar/'
        self.im_len = 60

    def _read_faces(self):
        f = open(self.dir_faces+'faces3_projections.csv','rb')
        self.faces = {}
        self.X = []
        self.persons = []

        for line in f:
            line = line.split('\n')[0]
            GT = line.split(',')[0]
            self.persons.append(GT)
            data = line.split(',')[1:]
            if GT not in self.faces:
                self.faces[GT] = []
            self.faces[GT].append(data)
            if self.X == []:
                self.X.append(data)
            else:
                self.X = np.vstack((self.X,data))

    def _read_faces_images(self):
        f = open(self.dir_faces+'faces_images.csv','rb')
        self.video_num = []
        self.images = []
        for line in f:
            line = line.split('\n')[0]
            vid = line.split('/')[1].split('_')[1]
            if vid not in self.video_num:
                self.video_num.append(int(vid))
            img = cv2.imread(self.dir_faces+line)
            self.images.append(cv2.resize(img, (self.im_len,self.im_len), interpolation = cv2.INTER_AREA))
            # cv2.imshow('test',img)
            # cv2.waitKey(100)
        print self.video_num

    def _cluster_faces(self):
        final_clf = 0
        best_v = 0

        for i in range(25):
            print '#####',i
            n_components_range = range(30, 65)
            cv_types = ['spherical', 'tied', 'diag', 'full']
            lowest_bic = np.infty
            # bic = []
            for cv_type in cv_types:
                for n_components in n_components_range:
                    gmm = mixture.GaussianMixture(n_components=n_components,
                                                  covariance_type=cv_type)
                    gmm.fit(self.X)
                    bic = gmm.bic(self.X)
                    if bic < lowest_bic:
                        lowest_bic = bic
                        best_gmm = gmm

            clf = best_gmm
            Y_ = clf.predict(self.X)

            v_meas = v_measure_score(self.persons, Y_)
            if v_meas > best_v:
                best_v = v_meas
                final_clf = clf
                # print final_clf.means_
                print best_v

        # print '----------------'
        self.best_v = best_v
        Y_ = final_clf.predict(self.X)

        self.predictions = {}
        for person in self.faces:
            self.predictions[person] = np.zeros(len(final_clf.means_))
            for i in self.faces[person]:
                self.predictions[person][final_clf.predict([i])[0]]+=1

        self.cost_matrix = np.zeros((len(self.faces),len(final_clf.means_)))

        for count,person in enumerate(self.predictions):
            self.predictions[person]/=np.sum(self.predictions[person])
            self.cost_matrix[count] = self.predictions[person]

        self.cluster_images = {}
        for img,val in zip(self.images,Y_):
            if val not in self.cluster_images:
                self.cluster_images[val] = []
            self.cluster_images[val].append(img)

        rows = 0
        maxi = 0
        for p in self.cluster_images:
            if len(self.cluster_images[p])>maxi:
                maxi = len(self.cluster_images[p])

        maxi = int(np.sqrt(maxi))
        for p in self.cluster_images:
            count = 0
            image = np.zeros((self.im_len*maxi,self.im_len*maxi,3),dtype=np.uint8)+255
            image_avg = np.zeros((self.im_len,self.im_len,3),dtype=np.uint8)
            for i in range(maxi):
                for j in range(maxi):
                    if count < len(self.cluster_images[p]):
                        image[i*self.im_len:(i+1)*self.im_len,j*self.im_len:(j+1)*self.im_len,:] = self.cluster_images[p][count]
                        image_avg += self.cluster_images[p][count]/(len(self.cluster_images[p])+1)
                    image[i*self.im_len:i*self.im_len+1,  j*self.im_len:(j+1)*self.im_len,  :] = 0
                    image[(i+1)*self.im_len:(i+1)*self.im_len+1,  j*self.im_len:(j+1)*self.im_len,  :] = 0
                    image[i*self.im_len:(i+1)*self.im_len,  j*self.im_len:j*self.im_len+1,  :] = 0
                    image[i*self.im_len:(i+1)*self.im_len,  (j+1)*self.im_len:(j+1)*self.im_len+1,  :] = 0
                    count+=1
            # cv2.imshow('test',image)
            # cv2.waitKey(2000)
            cv2.imwrite(self.dir_faces+str(p)+'.jpg',image)
            cv2.imwrite(self.dir_faces+str(p)+'_avg.jpg',image_avg)
        pickle.dump( [self.faces,final_clf,self.X,self.best_v], open( self.dir_faces+'faces_clusters.p', "wb" ) )

    def _assignment(self):
        pass

    def _read_tags(self):
        self.words_top = {}
        self.words_low = {}
        self.words_name = {}
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
            if i in self.video_num:
                self.words_name[i] = []
                for word in self.tags[i]['name']:
                    if str(word) not in self.words_name[i]:
                        self.words_name[i].append(str(word))
        print self.words_name


def main():
    f = faces_class()
    f._read_faces()
    f._read_faces_images()
    f._read_tags()

    # f._cluster_faces()
    # f._assignment()

if __name__=="__main__":
    main()
