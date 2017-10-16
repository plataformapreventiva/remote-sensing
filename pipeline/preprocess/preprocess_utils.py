import fastdtw as dtw #Distance function for time series
import gdalnumeric as gd
import numpy as np
from utils import *

# Functions for processing files, creating clusters, etc

def ts_distance(x,y,radius=2):
    """
    Compute Dynamic Time Warping Distance for two time series. Rule of thumb:
    the first element of X tells if the area is not valid (-2000 is Null value)
    """
    if x[0] < 0 or y[0] < 0:
        return 0.0
    else:
        dist, path = dtw.fastdtw(x, y, radius=radius)
        std1 = np.std(x)
        std2 = np.std(y)
        return dist/np.sqrt(std1*std2)


def pixel_distance_rows(stack):
    """
    Function for calculating the difference in behaviour between a pixel and its eastern neighbour. This function uses Dynamic Time Wraping to measure the difference between two time series in neighboring pixels.
    (stack): numpy stack in three dimensions sorted according to a) latitude, b) longitude, c) time
    Returns a numpy array with latitude, longitude and difference in behaviour with it's neighbor. The method eliminates one column.
    """

    # TODO: Find a more efficient way of iterating through a 2d array
    row, col, time = stack.shape
    mat = [[ts_distance(stack[i,j], stack[i,j+1]) for j in range(col-1)] for i in range(row)]
    return np.array(mat)

def pixel_distance_cols(stack):
"""
    Same as before, but with respect to a southern neighbor.
"""
    # TODO: Find a more efficient way of iterating through a 2d array
    row, col, time = stack.shape
    mat = [[ts_distance(stack[i,j], stack[i+1,j]) for j in range(col)] for i in range(row-1)]
    return np.array(mat)


def neighbor_clusters_rows(dist_array, cutoff):
    """
    Function for creating horizontal clusters according to neighboring data
    (dist_array): 2d array containing distances between a pixel and its eastern neighbor
    """
    rows, cols = dist_array.shape
    clust = []
    for i in range(rows):
        k, safe_k = 1, 1
        row = []
        for j in range(cols):
            row.append(cluster_value(dist_array[i,j], cutoff, k))
            k, safe_k = get_k(k, safe_k, row[j])
        clust.append(row)
    return np.array(clust)

def neighbor_clusters_cols(dist_array, cutoff):
    rows, cols = dist_array.shape
    clust = []
    for i in range(cols):
        k, safe_k = 1, 1
        row = []
        for j in range(rows):
            row.append(cluster_value(dist_array[j,i], cutoff, k))
            k, safe_k = get_k(k, safe_k, row[j])
        clust.append(row)
    return np.array(clust).transpose()

def get_k(k, safe_k, row_j):
    safe_k = k if k > 0 else safe_k
    k = row_j if row_j > 0 else safe_k
    return k, safe_k


def cluster_value(x, cutoff, k):
    if x < 0.0001:
        return 0
    elif x < cutoff:
        return k
    else:
        return k+1

def simple_cluster_join(h_mat, v_mat, cutoff, clust=None):
    """
    Sample function. Do not use, it creates clusters that are tilted by construction
    """
    rows, cols = h_mat.shape
    if clust is not None:
        clust = np.array([[None for col in range(cols+1)] for row in range(rows+1)])
    k = 1
    for i in range(rows):
        for j in range(cols):
            if h_mat[i, j] < 0.00000001 or v_mat[i, j] < 0.00000001:
                clust[i,j] = 0
            else:
                if clust[i,j]:
                    k = clust[i,j]
                    clust[i, j+1] = new_k(k, h_mat[i,j], cutoff, clust[i,j+1], v_mat[i-1,j-1])
                    clust[i+1, j] = k if v_mat[i,j] < cutoff else clust[i+1,j]
                else:
                    k = k + 1
                    clust[i,j] = k
                    clust[i, j+1] = new_k(k, h_mat[i,j], cutoff, clust[i,j+1], v_mat[i-1,j-1])
                    clust[i+1, j] = k if v_mat[i,j] < cutoff else clust[i+1,j]
    return clust[0:-1, 0:-1]



def cluster_join(h_clust, v_clust):
    """
    Fuction for creating 2d cluters given 1d clusters previously created (horizontally and vertically). This methods starts from the upper left corner of a map, and starts adding elements greedily to its cluster (just as a simple union).
    """
    rows, cols = h_clust.shape
    clust = np.array([[None for col in range(cols+1)] for row in range(rows+1)])
    k = 1
    ks = {k}
    for i in range(rows):
        for j in range(cols):
            print(i,j)
            # Do not process water elements
            if h_clust[i, j] < 0.00001 or v_clust[i, j] < 0.00001:
                clust[i,j] = 0
            else:
                # Obtain elements in row and column cluster
                row_clust = np.where(h_clust[i,:] == h_clust[i,j])[0]
                col_clust = np.where(v_clust[:,j] == v_clust[i,j])[0]
                # Obtain the values of the joint cluster for my cluster-neighbors
                # We will add all these to our current cluster (Yes, it's very greedy)
                row_clust_ref = set(clust[i, row_clust]).difference({None, 0})
                col_clust_ref = set(clust[col_clust,j]).difference({None, 0})
                clust_ref = row_clust_ref.union(col_clust_ref)
                # If the current pixel doenst belong to a cluster, create a new one
                self_clust = clust[i, j]
                if not self_clust:
                    self_clust, ks = new_cluster(ks, clust_ref)
                # Make all my cluster-neighbors my neighbors in the official cluster
                clust[i, row_clust] = self_clust
                clust[col_clust, j] = self_clust
                # And make their cluster friends my friends as well
                # Only if we're not friends already, ofc
                if len(clust_ref) > 1:
                    for ref in clust_ref:
                        if ref:
                            clust[np.where(clust == ref)] = self_clust
    return clust[0:-1, 0:-1]


def new_cluster(ks, clust_ref):
    ks = ks.difference(clust_ref) if clust_ref else ks
    max_ks = max(ks)
    new_k = min(ks.difference())
    new_k = max(ks) + 1 if ks else 1
    ks.add(new_k)
    return new_k, ks

def unify_clusters(clust_mat):
    clusters = set(clust_mat.flatten())
    i = 1
    try:
        clusters.remove(0)
    except:
        pass
    try:
        clusters.remove(None)
    except:
        pass
    for cluster in clusters:
        #print(cluster)
        where = np.where(clust_mat == cluster)
        if len(where[0]) < 2:
            clust_mat[where] = -1
        else:
            clust_mat[where] = i
            i = i+1
    clust_mat[np.where(clust_mat == -1)] = i

    return clust_mat


def new_k(k, self_val, cutoff, old_k, other_val):
    if old_k and other_val < self_val:
        return old_k
    else:
        if self_val < cutoff:
            return k
        else:
            return old_k
