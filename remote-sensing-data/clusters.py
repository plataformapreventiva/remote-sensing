# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os


mpos = ['01001','01002','01003'] #list of municipalities to search for

# read all the pickles
path = './remote-sensing-data/clusters/*.p'
pickles = glob.glob(path)

# find the pickles we want (the ones on mpos)
cls = []
for p in pickles:
        if any(x in p for x in mpos):
                cls.append(p)

im = []
for i in range(0,len(cls)):
        im.append(pd.read_pickle(pickles[i]).astype('int16'))

#im = im.astype('int16')
#plt.figure()
#plt.imshow(im)

nrows = int(len(cls)/5)

for t in range(0,len(cls)):
        image = im[t]
        pl = plt.subplot(nrows, 5, t+1)
        pl = plt.imshow(image, aspect='auto')#cmap = 'hot') #or cmap= 'nipy_spectral')
        pl.axes.get_xaxis().set_ticks([])
        pl.axes.get_yaxis().set_ticks([])
        


plt.show()
