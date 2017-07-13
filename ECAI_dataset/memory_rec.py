import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
from operator import add, mul

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# plt.style.use('fivethirtyeight')
# fig, ax = plt.subplots()
save_dir = "/home/omari/Datasets/scalibility/"
data = []
x = []
size = 0

plt.subplot(221)
plt.title('Extended Train Robots')
plt.xlabel('video')
plt.ylabel('bytes')
plt.grid(True)
plt.subplot(222)
plt.title('Leeds Activity Commands')
plt.xlabel('video')
plt.ylabel('bytes')
plt.grid(True)
plt.subplot(223)
plt.title('Extended Object Ordering')
plt.xlabel('video')
plt.ylabel('bytes')
plt.grid(True)
plt.subplot(224)
plt.title('Extended Kitchen Activities')
plt.xlabel('video')
plt.ylabel('bytes')
plt.grid(True)


#################################################################
## ECAI
#################################################################
plt.subplot(224)
# data = []
# x = []
# size = 0
# for i in range(1,494):
#     path = "/home/omari/Datasets/ECAI Data/dataset_segmented_15_12_16/vid"+str(i)
#     size += get_size(start_path=path)
#     data.append(size)
#     x.append(i)
# # print data
# pickle.dump( [x,data], open( save_dir+'ECAI_data.p', "wb" ) )
x1,data = pickle.load( open( save_dir+'ECAI_data.p', "rb" ) )

data = [1]+data
plt.semilogy(xrange(len(data)), data, linewidth=2.0)
plt.xlim([-10,len(data)+10])
plt.ylim([10**2,10**11])



#################################################################
## Baxter
#################################################################
plt.subplot(222)
# data = []
# x = []
# size = 0
# for i in range(1,205):
#     path = "/home/omari/Datasets/Baxter_Dataset_final/scene"+str(i)
#     size += get_size(start_path=path)
#     data.append(size)
#     x.append(i)
# # print data
# pickle.dump( [x,data], open( save_dir+'Baxter_data.p', "wb" ) )
x2,data = pickle.load( open( save_dir+'Baxter_data.p', "rb" ) )

##vision concepts
vision_no = []
for f in ["colours","shapes","locations","directions","distances"]:
    if vision_no == []:
        vision_no = pickle.load( open( save_dir+'Baxter/'+f+'_per_video.p', "rb" ) )
    else:
        vision_no = map(add, vision_no, pickle.load(open(save_dir+'Baxter/'+f+'_per_video.p',"rb")) )
vision = [x *126 for x in vision_no]        # the size of a single Gaussian model in hard disk

## groundings
ngrams = pickle.load( open( save_dir+'Baxter/ngrams_per_video.p', "rb" ) )
K = []
for i,j in zip(ngrams,vision_no):
    K.append(i*j)
grounding = [x *4 for x in K]               # the size of a single element in the K matrix

## grammar
grammar = [x*.1 for x in grounding]
print grammar

data = [1]+data
vision = [1]+vision
grounding = [1]+grounding
grammar = [1]+grammar
plt.semilogy(xrange(len(data)), data, linewidth=2.0, label='raw data')
plt.semilogy(xrange(len(data)), vision, linewidth=2.0, label='vision')
plt.semilogy(xrange(len(data)), grounding, linewidth=2.0, label='grounding')
plt.semilogy(xrange(len(data)), grammar, linewidth=2.0, label='grammar')
plt.xlim([-10,len(data)+10])
plt.ylim([10**2,10**11])


#################################################################
## Jivko
#################################################################
plt.subplot(223)
# data = []
# x = []
# size = 0
# i = 1
# for t in range(1,4):
#     for obj in range(1,33):
#         for act in ["drop","grasp","hold","lift","lower","press","push"]:
#             path = "/home/omari/Datasets/jivko_dataset/t"+str(t)+"/obj_"+str(obj)+"/trial_1/"+act
#             size += get_size(start_path=path)
#             print size
#             data.append(size)
#             x.append(i)
#             i+=1
# pickle.dump( [x,data], open( save_dir+'jivko_data.p', "wb" ) )
x3,data = pickle.load( open( save_dir+'jivko_data.p', "rb" ) )


data = [1]+data
plt.semilogy(xrange(len(data)), data, linewidth=2.0, label='raw data')
plt.semilogy(xrange(len(data)), data, linewidth=2.0, label='vision model')
plt.semilogy(xrange(len(data)), data, linewidth=2.0, label='grounding model')
plt.semilogy(xrange(len(data)), data, linewidth=2.0, label='grammar rules')
plt.xlim([-10,len(data)+10])
plt.ylim([10**2,10**11])

# Put a legend below current axis
plt.legend(loc='upper center', bbox_to_anchor=(1, -0.15),
          fancybox=True, shadow=True, ncol=5)

#################################################################
## Dukes
#################################################################
plt.subplot(221)
# data = []
# x = []
# size = 0
# for i in range(1,1001):
#     path = "/home/omari/Datasets/robot_modified/scenes/"+str(i)
#     size += get_size(start_path=path)
#     data.append(size)
#     x.append(i)
# # print data
# pickle.dump( [x,data], open( save_dir+'Dukes_data.p', "wb" ) )
x4,data = pickle.load( open( save_dir+'Dukes_data.p', "rb" ) )

data = [1]+data
plt.semilogy(xrange(len(data)), data, linewidth=2.0)
plt.xlim([-10,len(data)+10])
plt.ylim([10**2,10**11])



plt.show()
