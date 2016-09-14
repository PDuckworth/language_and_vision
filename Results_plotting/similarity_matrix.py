import numpy as np
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
# import matplotlib.pyplot as plt

font = FontProperties()
families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
font.set_family(families[0])

ax = plt.subplot(1, 3, 1)
matrix = np.random.rand(7,7)
plt.imshow(matrix, interpolation='nearest', cmap=plt.get_cmap('inferno'), extent=(0,7,0,7))
# plt.title('Real-world dataset', fontsize=20)
# plt.text(0.5, -.1, 'Real-world dataset',
#          horizontalalignment='center',
#          fontsize=20,
#          transform = ax.transAxes)
plt.xticks(np.arange(7)+.6, ['colour','shape','location','direction','distance','action','function'], fontsize=20, rotation=80)
ax.xaxis.tick_top()
plt.yticks(np.arange(7)+.5, reversed(['colour','shape','location','direction','distance','action','function']), fontsize=20)


ax = plt.subplot(1, 3, 2)
matrix = np.random.rand(7,7)
plt.imshow(matrix, interpolation='nearest', cmap=plt.get_cmap('inferno'), extent=(0,7,0,7))
# plt.xticks(np.arange(7)+.6, ['colour','shape','location','direction','distance','action','function'], fontsize=20, rotation=80)
ax.xaxis.tick_top()
# plt.yticks(np.arange(7)+.5, reversed(['colour','shape','location','direction','distance','action','function']), fontsize=20)


ax = plt.subplot(1, 3, 3)
matrix = np.random.rand(7,7)
plt.imshow(matrix, interpolation='nearest', cmap=plt.get_cmap('inferno'), extent=(0,7,0,7))
# plt.xticks(np.arange(7)+.6, ['colour','shape','location','direction','distance','action','function'], fontsize=20, rotation=80)
ax.xaxis.tick_top()
# plt.yticks(np.arange(7)+.5, reversed(['colour','shape','location','direction','distance','action','function']), fontsize=20)


# plt.colorbar()
plt.show()
