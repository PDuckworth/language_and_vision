#! /usr/bin/env python2.7
# -*- coding: iso-8859-1 -*-

from pylab import *
import matplotlib.font_manager as font_manager

t = arange(1,6)
s1 = [0,0,8,25,33]
s2 = [0,0,37,67,70]
s3 = [0,0,57,91,95]
s4 = [0,0,88,99,100]

#s1 = [0,100,100,100,100]
#s2 = [0,100,100,100,100]
#s3 = [0,0,60,87,92]
#s4 = [0,0,0,0,0]

#linema20, = plt.plot(t, s1, 'bo-', label='objects = 1')
#linema20, = plt.plot(t, s2, 'g^-', label='objects = 2')
#linema20, = plt.plot(t, s3, 'co-', lw=2, label='objects = 3')
#linema20, = plt.plot(t, s4, 'rd-', lw=2, label='objects = 4')

linema20, = plt.plot(t, s1, 'bo-', label='examples = 200')
linema20, = plt.plot(t, s2, 'g^-', label='examples = 500')
linema20, = plt.plot(t, s3, 'co-', label='examples = 1000')
linema20, = plt.plot(t, s4, 'rd-', label='examples = 2000')


xlabel('Number of Colors')
plt.axis([0, 5, -10, 110])
ylabel('Results')
title('Number of objects = 3')


props = font_manager.FontProperties(size=10)
leg = plt.legend(loc='center left', shadow=True, fancybox=True, prop=props)
leg.get_frame().set_alpha(0.5)

grid(True)
savefig("test.png")
show()
