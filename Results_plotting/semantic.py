# This is a ported version of a MATLAB example from the signal processing
# toolbox that showed some difference at one time between Matplotlib's and
# MATLAB's scaling of the PSD.  This differs from psd_demo3.py in that
# this uses a complex signal, so we can see that complex PSD's work properly
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


"""
Simple demo of a horizontal bar chart.
"""
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

plt.subplots_adjust(hspace=0.45, wspace=1.3)
ax = plt.subplot(3, 1, 1)
opacity = 0.8
bar_width=1

# index = np.arange(3)
# plt.xticks(index, ('A', 'B', 'C'))
# plt.title('real-world dataset')
# ax.bar([0], [0.2], color="blue", width=bar_width, label="A- unsupervised", alpha=opacity, align="center")
# ax.bar([1], [0.75], color="red", width=bar_width, label="B- our approach", alpha=opacity, align="center")
# ax.bar([2], [0.99], color="green", width=bar_width, label="C- supervised", alpha=opacity, align="center")
# ax.legend(loc=2)
#
# ax = plt.subplot(1, 2, 2)
# plt.xticks(index, ('A', 'B', 'C'))
# plt.title('synthetic-world dataset')
# ax.bar([0], [0.2], color="blue", width=bar_width, label="A- unsupervised", alpha=opacity, align="center")
# ax.bar([1], [0.88], color="red", width=bar_width, label="B- our approach", alpha=opacity, align="center")
# ax.bar([2], [0.99], color="green", width=bar_width, label="C- supervised", alpha=opacity, align="center")
# ax.legend(loc=2)

# Example data

people = ('unsupervised', 'our-system', 'supervised')
y_pos = np.arange(len(people))

plt.barh([0], [32.9], align='center', height=1, alpha=0.9,color='orange')
plt.barh([1], [85.6], align='center', height=1, alpha=0.7, color="green")
plt.barh([2], [98.1], align='center', height=1,  alpha=0.9, color=(.4,.3,1))

ax.text(32.9-4,0-.2,'32.9',size=16)
ax.text(85.6-4,1-.2,'85.6',size=16)
ax.text(98.1-4,2-.2,'98.1',size=16)

plt.xticks([0,20,40,60,80,100], ['0','20','40','60','80','100'], fontsize=20)
plt.yticks(y_pos, people, fontsize=20)
plt.title('Dukes (2013) dataset', fontsize=20)
plt.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')

ax = plt.subplot(3, 1, 2)

people = ('unsupervised', 'our-system', 'supervised')
y_pos = np.arange(len(people))

plt.barh([0], [39.8], align='center', height=1, alpha=0.9,color='orange')
plt.barh([1], [91.3], align='center', height=1, alpha=0.7, color="green")
plt.barh([2], [98.9], align='center', height=1, alpha=0.9, color=(.4,.3,1))

ax.text(31.2-4,0-.2,'39.8',size=16)
ax.text(81.5-4,1-.2,'91.3',size=16)
ax.text(97.4-4,2-.2,'98.9',size=16)

plt.xticks([0,20,40,60,80,100], ['0','20','40','60','80','100'], fontsize=20)
plt.yticks(y_pos, people, fontsize=20)
plt.title('Sinapov (2016) dataset', fontsize=20)
plt.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')


ax = plt.subplot(3, 1, 3)

people = ('unsupervised', 'our-system', 'supervised')
y_pos = np.arange(len(people))

plt.barh([0], [31.2], align='center', height=1, alpha=0.9,color='orange')
plt.barh([1], [81.5], align='center', height=1, alpha=0.7, color="green")
plt.barh([2], [97.4], align='center', height=1, alpha=0.9, color=(.4,.3,1))

ax.text(31.2-4,0-.2,'31.2',size=16)
ax.text(81.5-4,1-.2,'81.5',size=16)
ax.text(97.4-4,2-.2,'97.4',size=16)

plt.xticks([0,20,40,60,80,100], ['0','20','40','60','80','100'], fontsize=20)
plt.yticks(y_pos, people, fontsize=20)
plt.title('Alomari (2016) dataset', fontsize=20)
plt.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')

plt.show()
