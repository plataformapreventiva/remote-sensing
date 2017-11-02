import requests
import re
import os
import datetime
import subprocess, shlex
import pickle
import numpy as np
import gdal
import time
from gdalconst import *
from osgeo import osr
from bs4 import BeautifulSoup

# For processing files
import fastdtw as dtw #Distance function for time series
import gdalnumeric as gd
import numpy as np

import pdb


def create_tiles():
    '''
Simple function to generate all the tiles needed to cover Mexico
    '''
    h = {'h07', 'h08', 'h09'}
    v = {'v05', 'v06', 'v07'}
    ti = [h1+v1 for h1 in h for v1 in v]
    ti.remove('h09v05')
    return ti

def get_date_urls(modis_prod_url):
    '''
Given the url of a modis product, obtain all possible dates
e.g.: mods_prod_url = "https://e4ftl01.cr.usgs.gov/MOLT/MOD13Q1.006/"
    '''
    r = requests.get(modis_prod_url)
    if r:
        soup = BeautifulSoup(r.text, 'html.parser')
        re_date = '[0-9]{4}\.[0-9]{2}\.[0-9]{2}'
        return [link.get('href') for link in soup.find_all('a') if re.search(re_date, link.text)]

    else:
        return []

def get_valid_dates(start, end, modis_prod_url='https://e4ftl01.cr.usgs.gov/MOLT/MOD13Q1.006/'):

    """
    Function to obtain valid MODIS dates for a specific product, where
    start: start date in format yyyy.mm.dd
    end: end date in format yyyy.mm.dd
    modis_prod_url: from which to get dates
    NOTE: THIS IS VERY LAZY AND INEFFICIENT, DO REVISIT, and get an ordered set
    """
    s_date = datetime.datetime.strptime(start, '%Y.%m.%d')
    e_date = datetime.datetime.strptime(end, '%Y.%m.%d')
    dates = [s_date + datetime.timedelta(days=x) for x in range(0, (e_date - s_date).days)]
    try:
        u_dates = get_date_urls(modis_prod_url)
    except:
        with open('available_dates.p', 'r') as pck:
            u_dates = pickle.load(pck)
    #TODO: check if this fails or not

    u_dates_dt = [datetime.datetime.strptime(x, '%Y.%m.%d/') for x in u_dates]
    valid_dates = list(set(u_dates_dt).intersection(dates))
    valid_dates.sort()

    return [date.strftime('%Y.%m.%d') for date in valid_dates]


def valid_path(path):
    """
    Create path if it doesn't exist
    """
    if not os.path.exists(path):
        os.makedirs(path)
    pass

def check_empty_file(path):
    """
    Check if a file has zero bytes, if so, erase it. 
    """
    try: 
    	statinfo = os.stat(path).st_size
    except OSError:
	statinfo = True
    if not statinfo:
        os.remove(path)



def get_tile_urls(modis_prod_url, date_url, tiles):
    '''
Given tiles from create_tiles and date_url from get_date_urls, obtain the specific filename for the tiles to download
    '''

    r = requests.get(modis_prod_url + date_url)
    if r:
        soup = BeautifulSoup(r.text, 'html.parser')
        a = [link.get('href') for link in soup.find_all('a')]
        tile_urls= [filestr for filestr in a for tile in tiles if re.search(tile+'.*\.hdf$', filestr)]
        return full_tile_urls(modis_prod_url, date_url, tile_urls)
    else:
        return []

def full_tile_urls(modis_prod_url, date_url, tile_urls):
    '''
Given date_url from get_date_urls, and tile file names from get_tile_urls, obtain the full url to download the file
    '''
    if date_url[-1] != '/':
        date_url = date_url + '/'
    return [modis_prod_url + date_url + tile_url for tile_url in tile_urls]

def get_suffix(modis_prod_url):
    prod = re.search('(MOD\w+\.\d{3})', modis_prod_url)
    suff_dict = {'MOD13A2.006':'":MODIS_Grid_16DAY_1km_VI:1\ km\ 16\ days\ EVI',
            'MOD13Q1.006':'":MODIS_Grid_16DAY_250m_500m_VI:250m\ 16\ days\ EVI'}
    if prod:
        try:
            return suff_dict[prod.group(0)]
        except KeyError:
            print('---    UNKNOWN MODIS PRODUCT   ---')
            print('Add a new product on function get_suffix in tile_url_utils.py')
    else:
        print('--- MODIS PRODUCT NOT RECOGNIZED ---')
        print('Add a new product on function get_suffix in tile_url_utils.py')


def isnone(x):
    if x is None:
        return True
    else:
        return False

def date_rasternames(main_path, dates, dimension='', cve_muni='', cutoff=''):
    if cve_muni:
        cve_muni = cve_muni + '_'
    if dimension:
        dimension = '_' + dimension
    if cutoff:
        cutoff = '_' + cutoff
    return main_path + cve_muni + dates[0] + '_' + dates[-1] + dimension + cutoff  +'.tiff'


############################################
#### GDAL FUNTCTIONS
############################################

# Function to read the original file's projection:
def get_geo_info(FileName):
    """
    Function for obtaining information from Raster files|

    """
    SourceDS = gdal.Open(FileName, GA_ReadOnly)
    NDV = SourceDS.GetRasterBand(1).GetNoDataValue()
    xsize = SourceDS.RasterXSize
    ysize = SourceDS.RasterYSize
    GeoT = SourceDS.GetGeoTransform()
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    DataType = SourceDS.GetRasterBand(1).DataType
    DataType = gdal.GetDataTypeName(DataType)
    return NDV, xsize, ysize, GeoT, Projection, DataType


# Function to write a new file.
def create_geotiff(NewFileName, Array, driver, NDV,
                  xsize, ysize, GeoT, Projection, DataType):
    """
    Function for writing a spatial file
    """
    if DataType == 'Int16':
        DataType = gdal.GDT_Int16
    else:
        try:
            DataType = eval('gdal.GDT_' + DataType)
        except AttributeError:
            raise('DataType not recognized')
    # Set nans to the original No Data Value
    Array[isnone(Array)] = NDV
    # Set up the dataset
    DataSet = driver.Create(NewFileName, xsize, ysize, 1, DataType)
            # the '1' is for band 1.
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection(Projection.ExportToWkt())
    # Write the array
    DataSet.GetRasterBand(1).WriteArray(Array)
    DataSet.GetRasterBand(1).SetNoDataValue(NDV)
    return NewFileName

def write_raster(Array, FileName, ReferenceName, xsize=None, ysize=None, DataType=None):
    """
    Write raster file with specifications according to a reference raster.
    (Array): Numpy Array to Write
    (FileName): String output file name
    (ReferenceName): String reference file name
    """

    NDV, xsizeO, ysizeO, GeoT, Projection, DataType0 = get_geo_info(ReferenceName)
    if not xsize:
        xsize = xsizeO
    if not ysize:
        ysize = ysizeO
    ysize, xsize = Array.shape
    if not DataType:
        DataType = DataType0
    driver = gdal.GetDriverByName('GTiff')
    return create_geotiff(FileName, Array, driver, NDV, xsize, ysize, GeoT, Projection, DataType)


def create_mun_shp(mun, shp_dest, shp_orig='../shp/muns.shp'):
    cmd_list = ["ogr2ogr -where", '"CVE_MUNI='+ "'" + mun + "'" + '"', shp_dest, shp_orig]
    cmd = ' '.join(cmd_list)
    subprocess.Popen(cmd, shell=True)
    return True


def crop_raster(shp_path, in_raster, out_raster):
    cmd_list = ['gdalwarp', '-cutline', shp_path, '-crop_to_cutline', '-dstnodata -2000', in_raster, out_raster]
    cmd = " ".join(cmd_list)
    subprocess.call(cmd, shell=True)

def get_mun_raster(mun, in_raster, out_raster, mun_shp='../shp/muns.shp'):
    valid_path("tmp_mun_shp")
    tmp_path = "tmp_mun_shp/" + mun + ".shp"
    t = create_mun_shp(mun, tmp_path, mun_shp)
    time.sleep(1)
    crop_raster(tmp_path, in_raster, out_raster)

def rasters_stack(dates, path, suffix):
    rasters = [gd.LoadFile(path + date + suffix) for date in dates]
    return np.dstack(rasters)

def raster_to_numpy(path):
    return gd.LoadFile(path)

################################################
#### FUNCTIONS FOR PROCESSING
################################################

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
        print(cluster)
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


