# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import fastdtw as dtw
from pydtw import dtw1d
import utils


# ==============================================================
# ========== generate some initial data  =======================
# ==============================================================
# t = 50  # sample rate = 100
# waves = 4  # length f the signal
# x = np.arange(0, 2*math.pi*waves, 2*math.pi*(1/t))
# alpha = math.pi*0.05
# A = [math.cos(i) for i in x]
# B = [math.cos(i + alpha) + 3*alpha for i in x]
# Cc = [math.cos(i + 2*alpha) for i in x]
# D = [math.cos(i + 3*alpha) for i in x]
# E = [math.cos(i + 4*alpha) for i in x]
# F = [math.cos(i + 5*alpha) for i in x]
# initTS = [0 for i in x]
# C = initTS.copy()
# Cprime = [0 for k in range(0,len(initTS))]
# S = [A, B, Cc, D, E, F]
# ==============================================================
# ==============================================================



def DBA (C,S):
    ''' Function to compute the DTW Barycenter Averaging. fastdtw required

    Parameters:
        C - List containing initial series of averages.
        S - List of lists containing the set of series to average
            S = [S_1, S_2, .. , S_n ]; and Si = [s_i1, ..., s_in_i]

    Output:
        Cprime - Computed list containing the new average using the DBA algorithm.

    REFERENCE: ALGORITHM 5 from:
               Petitjean, et.al (2011), A global averaging method for dynamic time warping, with applications to clustering
    '''
    T = len(S[1])
    Tprime = len(C)
    assoctab = [[] for k in range(Tprime)]
    for seq in range(0, len(S)):
        dist, path = dtw.fastdtw(C, S[seq], radius=len(S[seq]))
        lon = len(path) - 1
        while lon >= -1:  #  i > -1 and j > -1):
            assoctab[path[lon][0]].append(S[seq][path[lon][1]])
            lon = lon - 1

    for i in range(0,T):
        Cprime[i] = np.mean(assoctab[i])  # for the author od the article the barycenter is the arithmetic mean.
    return Cprime

""" Space to play: 

plt.plot(x, S[0], label='cos(t)')
plt.plot(x, S[1], label='cos(t+alpha)')
plt.plot(x, S[2], label='cos(t+2alpha)')
#plt.plot(x, S[3], label='cos(t+3alpha)')
#plt.plot(x, S[4], label='cos(t+4alpha)')
#plt.plot(x, S[5], label='cos(t+5alpha)')

plt.plot(x, initTS, label='initial TS')

Cprime = DBA(C, S)
#plt.plot(x, Cprime, label='average iter'+str(1))
for k in range(1, 5):
    C = Cprime.copy()
    Cprime = DBA(C, S)
    #plt.plot(x, Cprime, label='average iter'+str(k+1))
plt.plot(x, Cprime, label='average iter'+str(k+1))
plt.legend()
plt.show()

"""




