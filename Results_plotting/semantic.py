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

plt.barh([0], [14], align='center', height=1, alpha=0.4)
plt.barh([1], [75], align='center', height=1, alpha=0.4, color="red")
plt.barh([2], [99], align='center', height=1, alpha=0.4, color="green")
plt.xticks([0,20,40,60,80,100], ['0','20','40','60','80','100'], fontsize=20)
plt.yticks(y_pos, people, fontsize=20)
plt.title('Alomari (2016) dataset', fontsize=20)
plt.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')

ax = plt.subplot(3, 1, 2)

people = ('unsupervised', 'our-system', 'supervised')
y_pos = np.arange(len(people))

plt.barh([0], [14], align='center', height=1, alpha=0.4)
plt.barh([1], [75], align='center', height=1, alpha=0.4, color="red")
plt.barh([2], [99], align='center', height=1, alpha=0.4, color="green")
plt.xticks([0,20,40,60,80,100], ['0','20','40','60','80','100'], fontsize=20)
plt.yticks(y_pos, people, fontsize=20)
plt.title('Jivko (2016) dataset', fontsize=20)
plt.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')

ax = plt.subplot(3, 1, 3)

people = ('unsupervised', 'our-system', 'supervised')
y_pos = np.arange(len(people))

plt.barh([0], [22], align='center', height=1, alpha=0.4)
plt.barh([1], [88], align='center', height=1, alpha=0.4, color="red")
plt.barh([2], [99], align='center', height=1, alpha=0.4, color="green")
plt.xticks([0,20,40,60,80,100], ['0','20','40','60','80','100'], fontsize=20)
plt.yticks(y_pos, people, fontsize=20)
plt.title('Dukes (2013) dataset', fontsize=20)
plt.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')

plt.show()
