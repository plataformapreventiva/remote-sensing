# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import fastdtw as dtw
from pydtw import dtw1d



def minVal(v1, v2, v3):
    '''
    #===============================================================================
    # function to return the min of of 3 vectors based on the first entry comparison
    #===============================================================================
    '''
    if v1[0] <= np.min([v2[0], v3[0]]):
        return v1
    else:
        if v2[0] <= v3[0]:
            return v2
        else:
            return v3

def DTW(A,B):
    '''
    #=====================================================================================================
    # function to return  the Dynamic Time Warp in a 3 dimensional array m_TS
    #   input: A, B - time series in 2 dimensions
    #
    #   output: m_TS - 3-dimensional array of (distance to reach that node, path(x_coord), path(y_coord) )
    #
    # REFERENCE: to ALGORITHM 1 from:
    #           Petitjean, et.al (2011), A global averaging method for dynamic time warping, with applications to clustering
    #
    #=====================================================================================================

    # ======================================================
    # ========= generate sample data  ======================
    # ======================================================
    t = 5 # sample rate = 100
    waves = 1 # frequency of the singal
    x = np.arange(0, 2*math.pi*waves,2*math.pi*(1/t))
    alpha = math.pi*0.25
    A = [math.cos(i) for i in x]
    plt.plot(x, A, label = 'cos(t)')
    B = [math.cos(i +alpha) for i in x]
    plt.plot(x, B, label='cos(t+alpha)')
    C = [i +alpha*1.5 for i in B]
    plt.plot(x, C, label='cos(t+alpha)+alpha')
    plt.legend()
    plt.show()
    # ============================================================
    '''
    S = len(A)
    T = len(B)
    m_ST = np.ones([S, T, 3])  # matrix of couples (cost,path)
    # compute 1st entrance:
    m_ST[0, 0, 0] = np.abs(A[0] - B[0])  # probar con np.abs(a ver si es más rápido)
    m_ST[0, 0, 1:3] = [0, 0]  # the path from 1st node is (0,0)

    #  fill first row
    for i in range(1, S):
        m_ST[i, 0, 0] = np.abs(A[i] - B[0]) + m_ST[i-1, 0, 0]
        m_ST[i, 0, 1:3] = [i-1, 0]
    #  fill first column
    for j in range(1, T):
        m_ST[0, j, 0] = np.abs(A[0] - B[j]) + m_ST[0, j-1, 0]
        m_ST[0, j, 1:3] = [0, j-1]

    for i in range(1, S):
        for j in range(1, T):
            minimum = minVal(m_ST[i-1, j], m_ST[i, j-1], m_ST[i-1, j-1])
            m_ST[i, j, 0] = minimum[0] + np.abs(A[i]-B[j])
            m_ST[i, j, 1:3] = minimum[1:3]

    return m_ST


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
    T = len(S[0])
    Tprime = len(C)
    assoctab = [[] for k in range(Tprime)]
    for seq in range(0, len(S)):
        dist, path = dtw.fastdtw(C, S[seq], radius=len(S[seq]))
        lon = len(path) - 1
        while lon >= -1:  #  i > -1 and j > -1):
            assoctab[path[lon][0]].append(S[seq][path[lon][1]])
            lon = lon - 1

    Cprime = np.zeros(C.shape)
    for i in range(0,T):
        Cprime[i] = np.mean(assoctab[i])  # for the author of the article the barycenter is the arithmetic mean.
    return Cprime


