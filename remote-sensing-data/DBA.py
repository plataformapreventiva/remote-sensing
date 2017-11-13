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
t = 50  # sample rate = 100
waves = 1  # length f the signal
x = np.arange(0, 2*math.pi*waves, 2*math.pi*(1/t))
alpha = math.pi*0.05
A = [math.cos(i) for i in x]
B = [math.cos(i + 0*alpha)+2*alpha for i in x]
#Cc = [math.cos(i + 2*alpha) for i in x]
#D = [math.cos(i + 3*alpha) for i in x]
#E = [math.cos(i + 4*alpha) for i in x]
#F = [math.cos(i + 5*alpha) for i in x]
initTS = [0 for i in x]
# ==============================================================
# ==============================================================

C = initTS.copy()
Cprime = [0 for k in range(0,len(initTS))]
S = [A, B]#, Cc, D, E, F]

def DBA (C,S):
    T = len(S[1])
    Tprime = len(C)
    assoctab = [[] for k in range(Tprime)]
    for seq in range(0, len(S)):
        dist, path = dtw.fastdtw(S[seq], C, radius=2)
        #i = Tprime - 1; j = T - 1
        long = len(path) - 1
        while long >= -1:  #  i > -1 and j > -1):
            assoctab[path[long][0]].append(S[seq][path[long][1]])
            long = long - 1
            #assoctab[i].append(S[seq][j])
            #i, j = path[l]

    for i in range(0,T):
        Cprime[i] = np.mean(assoctab[i])  # for the author od the article the barycenter is the arithmetic mean.
    return Cprime


plt.plot(x, S[0], label='cos(t)')
plt.plot(x, S[1], label='cos(t+alpha)')
#plt.plot(x, S[2], label='cos(t+2alpha)')
#plt.plot(x, S[3], label='cos(t+3alpha)')
#plt.plot(x, S[4], label='cos(t+4alpha)')
#plt.plot(x, S[5], label='cos(t+5alpha)')

plt.plot(x, initTS, label='initial TS')

Cprime = DBA(C, S)
plt.plot(x, Cprime, label='average iter'+str(1))
for k in range(1,100):
    C = Cprime.copy()
    Cprime = DBA(C, S)
    #plt.plot(x, Cprime, label='average iter'+str(k+1))
plt.plot(x, Cprime, label='average iter'+str(k+1))
plt.legend()
plt.show()
