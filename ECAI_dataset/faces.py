import numpy as np
from sklearn import mixture
import itertools
from sklearn.metrics.cluster import v_measure_score
import cv2

class faces_class():
    """docstring for faces"""
    def __init__(self):
        self.dir_faces =  '/home/omari/Datasets_old/ECAI_dataset_segmented/faces/'
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
        self.images = []
        for line in f:
            line = line.split('\n')[0]
            img = cv2.imread(self.dir_faces+line)
            self.images.append(cv2.resize(img, (self.im_len,self.im_len), interpolation = cv2.INTER_AREA))
            # cv2.imshow('test',img)
            # cv2.waitKey(100)

    def _cluster_faces(self):
        final_clf = 0
        best_v = 0

        for i in range(25):
            print '#####',i
            n_components_range = range(28, 35)
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
            self.predictions[person] = np.zeros(len(clf.means_))
            for i in self.faces[person]:
                self.predictions[person][clf.predict([i])[0]]+=1

        self.cost_matrix = np.zeros((len(self.faces),len(clf.means_)))

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


    def _assignment(self):
        # print np.max(self.cost_matrix)
        # print self.cost_matrix.argmax(axis=0)
        # print '######################################r'
        # print self.cost_matrix
        # indices = np.where(self.cost_matrix == self.cost_matrix.max())
        # print indices

        # name = '/results_%s.txt' % percentage
        f1 = open(self.dir_faces+'results.txt', 'w')
        # f1.write('n_topics: %s \n' % n_topics)
        # f1.write('n_iters: %s \n' % n_iters)
        # f1.write('dirichlet_params: (%s, %s) \n' % (dirichlet_params[0], dirichlet_params[1]))
        # f1.write('class_thresh: %s \n' % class_thresh)
        # f1.write('code book length: %s \n' % codebook_lengh)
        # f1.write('sum of all words: %s \n' % sum_of_all_words)
        # f1.write('videos classified: %s \n \n' % len(pred_labels))

        f1.write('v-score: %s \n' % self.best_v)
        # f1.write('homo-score: %s \n' % 0)
        # f1.write('comp-score: %s \n' % 0)
        # f1.write('mi: %s \n' % mi)
        # f1.write('nmi: %s \n \n' % nmi)
        # f1.write('mat: \n')

        # headings = ['{:3d}'.format(int(r)) for r in xrange(n_topics)]
        # f1.write('T = %s \n \n' % headings)
        # for row, lab in zip(mat, labs):
        #     text_row = ['{:3d}'.format(int(r)) for r in row]
        #     f1.write('    %s : %s \n' % (text_row, lab))
        # f1.write('\n')
        # f1.write('relevant_words: \n')
        # for i, words in relevant_words[percentage].items():
        #     f1.write('Topic %s : %s \n' % (i, words[:10]))
        # f1.close()



def main():
    f = faces_class()
    f._read_faces()
    f._read_faces_images()
    f._cluster_faces()
    # f._assignment()

if __name__=="__main__":
    main()
