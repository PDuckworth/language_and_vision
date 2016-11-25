import numpy as np
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import normalize

# import matplotlib.pyplot as plt

font = FontProperties()
families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
font.set_family(families[0])

ax = plt.subplot(3, 3, 1)
matrix = np.random.rand(7,7)
matrix = [[.98,0,0,0,0,0,.02],
            [0,1,0,0,0,0,0],
            [0,0,.9,0,0,0,.1],
            [0,0,0,.95,0,0,.05],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]]
plt.imshow(matrix, interpolation='nearest', extent=(0,7,0,7), clim=(0.0, 1.0))
plt.xticks(np.arange(7)+.6, ['','','','','','',''], fontsize=20, rotation=80)
ax.xaxis.tick_top()
ax.yaxis.tick_left()
plt.yticks(np.arange(7)+.5, reversed(['','','','','','','']), fontsize=20)


ax = plt.subplot(3, 3, 2)
matrix = np.random.rand(7,7)
matrix = [[.89,0,0,0,0,0,.11],
            [0,.91,0,0,0,0,.09],
            [0,0,.7,.2,0,0,.1],
            [0,0,0,.85,0,0,.15],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,.9,.1],
            [.02,0.03,.05,.04,0,.05,.8]]
plt.imshow(matrix, interpolation='nearest', extent=(0,7,0,7), clim=(0.0, 1.0))

#cmap=plt.get_cmap('inferno')
plt.xticks(np.arange(7)+.6, ['','','','','','',''], fontsize=20, rotation=80)
ax.xaxis.tick_top()
ax.yaxis.tick_left()
plt.yticks(np.arange(7)+.5, reversed(['','','','','','','']), fontsize=20)


ax = plt.subplot(3, 3, 3)
matrix = np.random.rand(7,7)
# print matrix
matrix = [[ 0.30252984,0.24621819,0.53444678,0.09301509,0.99190019,0.15002601
, 0.41706664],
 [ 0.31080874,0.00201452,0.92920843,0.57361588,0.51353934,0.09638897
, 0.24036975],
 [ 0.69559798,0.82576623,0.17220618,0.2091397, 0.03824165,0.20508842
, 0.33877414],
 [ 0.63346481,0.72486516,0.54593934,0.79550314,0.64845307,0.81904787
, 0.35893592],
 [ 0.68264223,0.82080099,0.4184158, 0.93548225,0.56862927,0.08776433
, 0.29677968],
 [ 0.9354688, 0.47913394,0.50191662,0.21305136,0.55621291,0.4677188
, 0.42924107],
 [ 0.04532964,0.17261178,0.97722363,0.97822649,0.79029184,0.45882575
, 0.96776001]]

matrix2 = np.random.rand(7,7)
for c,i in enumerate(matrix):
    matrix2[c] = i/np.sum(i)
reordering = np.argmax(matrix2, axis=1)
reordered = matrix2[:,reordering]

reordered[4] = [0,0,0,0,0,0,0]
reordered[:,4] = [0,0,0,0,0,0,0]

print '>>>',reordered
plt.imshow(reordered, interpolation='nearest', extent=(0,7,0,7), clim=(0.0, 1.0))

plt.xticks(np.arange(7)+.6, ['','','','','','',''], fontsize=20, rotation=80)
ax.xaxis.tick_top()
ax.yaxis.tick_left()
plt.yticks(np.arange(7)+.5, reversed(['','','','','','','']), fontsize=20)

################
## Jivko
###############


ax = plt.subplot(3, 3, 4)
matrix = np.random.rand(7,7)
matrix = [[.94,0,0,0,0,0,.02],
            [0,.96,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,.95,0,0,.05],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]]
plt.imshow(matrix, interpolation='nearest', extent=(0,7,0,7), clim=(0.0, 1.0))
plt.xticks(np.arange(7)+.6, ['','','','','','',''], fontsize=20, rotation=80)
ax.xaxis.tick_top()
ax.yaxis.tick_left()
plt.yticks(np.arange(7)+.5, reversed(['','','','','','','']), fontsize=20)


ax = plt.subplot(3, 3, 5)
matrix = np.random.rand(7,7)
matrix = [[.89,0,0,0,0,0,.11],
            [0,.91,0,0,0,0,.09],
            [0,0,0,0,0,0,0],
            [0,0,0,.85,0,0,.15],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,.9,.1],
            [.03,0.02,.0,.04,0,.01,.9]]
plt.imshow(matrix, interpolation='nearest', extent=(0,7,0,7), clim=(0.0, 1.0))

#cmap=plt.get_cmap('inferno')
plt.xticks(np.arange(7)+.6, ['','','','','','',''], fontsize=20, rotation=80)
ax.xaxis.tick_top()
ax.yaxis.tick_left()
plt.yticks(np.arange(7)+.5, reversed(['','','','','','','']), fontsize=20)


ax = plt.subplot(3, 3, 6)
matrix = np.random.rand(7,7)
for c,i in enumerate(matrix):
    matrix[c,c]*=4

print matrix

# matrix = [[ 0.30252984,0.24621819,0.53444678,0.09301509,0.99190019,0.15002601
# , 0.41706664],
#  [ 0.31080874,0.00201452,0.92920843,0.57361588,0.51353934,0.09638897
# , 0.24036975],
#  [ 0.69559798,0.82576623,0.17220618,0.2091397, 0.03824165,0.20508842
# , 0.33877414],
#  [ 0.63346481,0.72486516,0.54593934,0.79550314,0.64845307,0.81904787
# , 0.35893592],
#  [ 0.68264223,0.82080099,0.4184158, 0.93548225,0.56862927,0.08776433
# , 0.29677968],
#  [ 0.9354688, 0.47913394,0.50191662,0.21305136,0.55621291,0.4677188
# , 0.42924107],
#  [ 0.04532964,0.17261178,0.97722363,0.97822649,0.79029184,0.45882575
# , 0.96776001]]

matrix2 = np.random.rand(7,7)
for c,i in enumerate(matrix):
    matrix2[c] = i/np.sum(i)
reordering = np.argmax(matrix2, axis=1)
reordered = matrix2[:,reordering]

reordered[4] = [0,0,0,0,0,0,0]
reordered[:,4] = [0,0,0,0,0,0,0]

reordered[2] = [0,0,0,0,0,0,0]
reordered[:,2] = [0,0,0,0,0,0,0]

print '>>>',reordered
plt.imshow(reordered, interpolation='nearest', extent=(0,7,0,7), clim=(0.0, 1.0))

plt.xticks(np.arange(7)+.6, ['','','','','','',''], fontsize=20, rotation=80)
ax.xaxis.tick_top()
ax.yaxis.tick_left()
plt.yticks(np.arange(7)+.5, reversed(['','','','','','','']), fontsize=20)




ax = plt.subplot(3, 3, 7)
matrix = np.random.rand(7,7)
matrix = [[.96,0,0,0,0,0,.04],
            [0,.95,0,0,0,0,.05],
            [0,0,.9,0,0,0,.1],
            [0,0,0,.95,0,0,.05],
            [0,0,0,0,.93,0,.07],
            [0,0,0,0,0,.98,.02],
            [0,0,0,0,0,.02,.98]]
plt.imshow(matrix, interpolation='nearest', extent=(0,7,0,7), clim=(0.0, 1.0))
plt.xticks(np.arange(7)+.6, ['','','','','','',''], fontsize=20, rotation=80)
ax.xaxis.tick_top()
ax.yaxis.tick_left()
plt.yticks(np.arange(7)+.5, reversed(['','','','','','','']), fontsize=20)


ax = plt.subplot(3, 3, 8)
matrix = np.random.rand(7,7)
matrix = [[.81,0,0,0,0,0,.19],
            [0,.85,0,0,0,0,.15],
            [0,0,.76,.11,0,0,.13],
            [0,0,0,.8,0,0,.2],
            [0,0,0,0,.84,0,.16],
            [0,0,0,0,0,.9,.1],
            [.02,0.02,.01,.04,0,.05,.85]]
plt.imshow(matrix, interpolation='nearest', extent=(0,7,0,7), clim=(0.0, 1.0))

#cmap=plt.get_cmap('inferno')
plt.xticks(np.arange(7)+.6, ['','','','','','',''], fontsize=20, rotation=80)
ax.xaxis.tick_top()
ax.yaxis.tick_left()
plt.yticks(np.arange(7)+.5, reversed(['','','','','','','']), fontsize=20)


ax = plt.subplot(3, 3, 9)
matrix = np.random.rand(7,7)
# print matrix
# matrix = [[ 0.30252984,0.24621819,0.53444678,0.09301509,0.99190019,0.15002601
# , 0.41706664],
#  [ 0.31080874,0.00201452,0.92920843,0.57361588,0.51353934,0.09638897
# , 0.24036975],
#  [ 0.69559798,0.82576623,0.17220618,0.2091397, 0.03824165,0.20508842
# , 0.33877414],
#  [ 0.63346481,0.72486516,0.54593934,0.79550314,0.64845307,0.81904787
# , 0.35893592],
#  [ 0.68264223,0.82080099,0.4184158, 0.93548225,0.56862927,0.08776433
# , 0.29677968],
#  [ 0.9354688, 0.47913394,0.50191662,0.21305136,0.55621291,0.4677188
# , 0.42924107],
#  [ 0.04532964,0.17261178,0.97722363,0.97822649,0.79029184,0.45882575
# , 0.96776001]]

matrix2 = np.random.rand(7,7)
for c,i in enumerate(matrix):
    matrix[c,c]*=4
for c,i in enumerate(matrix):
    matrix2[c] = i/np.sum(i)
    # print np.sum(matrix2[c])
reordering = np.argmax(matrix2, axis=1)
reordered = matrix2[:,reordering]
# reordered[4] = [0,0,0,0,0,0,0]
# reordered[:,4] = [0,0,0,0,0,0,0]

plt.imshow(reordered, interpolation='nearest', extent=(0,7,0,7), clim=(0.0, 1.0))

plt.xticks(np.arange(7)+.6, ['','','','','','',''], fontsize=20, rotation=80)
ax.xaxis.tick_top()
ax.yaxis.tick_left()
plt.yticks(np.arange(7)+.5, reversed(['','','','','','','']), fontsize=20)



plt.colorbar()
plt.show()
