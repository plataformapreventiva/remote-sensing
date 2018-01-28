# -*- coding: utf-8 -*-
"""
Script for forecasting the NDVI observations using a Long Short Term Memory (LSTM) NN  
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import math
#import datetime
import os
##############
os.chdir("C:\\Users\\Raul Vazquez\\Desktop\\NNproj\\remote-sensing\\remote-sensing-data")
%matplotlib auto
##############
import utils_loc
from tf_lstm import lstm_model

#import glob

# import NDVI data
cve_mpo, ndvi_TS, ndvi_clust, init_date, end_date = utils_loc.read_NDVI(mun_num=6, clust_cut_val=2)
init_date = np.datetime64(init_date)
end_date = np.datetime64(end_date)
# visualize the TS
utils_loc.gridPlotsTS(ndvi_TS, cve_mpo)





y_TPV, y_RPV, y_TOI, y_ROI = utils_loc.read_agricolaDB(cve_mpo, init_date, end_date)

# VISUALIZE the y_series:
try:
    plt.figure()
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())# .MonthLocator())
    if not y_RPV.empty: plt.plot(y_RPV.index.get_values(),y_RPV['prop_cosecha'], label = 'riego primavera-verano')
    if not y_TPV.empty: plt.plot(y_TPV.index.get_values(), y_TPV['prop_cosecha'], label = 'temporal primavera-verano')
    if not y_ROI.empty: plt.plot(y_ROI.index.get_values(),y_ROI['prop_cosecha'], label = 'riego otono-invierno')
    if not y_TOI.empty: plt.plot(y_TOI.index.get_values(), y_TOI['prop_cosecha'], label = 'temporal otono-invierno')
    plt.title('Proporción de superficie cosechada respecto a la sembrada en el municipio ' + cve_mpo)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) # gives the date format defined in xmft
    plt.gcf().autofmt_xdate() # tilts the x labels
    plt.legend()
    plt.show()
    plt.xticks(pd.date_range( (max(np.datetime64(init_date),
                             y_TPV.index.get_values()[0]) -1) ,end_date, freq='1M') )
    #plt.xticks( y_TPV.index.get_values() )
except:
        pass


'''
MODIS release images every 16 days
    1 year contains 23 observations s.t. every year the observations are on the same date 
    (last observation of the year contains less days)
'''
ndvi_obsDates = [np.arange( (init_date + np.timedelta64((365+1)*aa,'D')), 
                            (init_date + np.timedelta64((365+1)*aa,'D') + np.timedelta64(357,'D') ),
                             np.timedelta64(16,'D'))
                for aa in range(math.ceil( (end_date-init_date)/np.timedelta64(365,'D')  ))]
ndvi_obsDates = np.array(ndvi_obsDates).flatten()


''' ------- PREPROCESS X variable ---------- '''

# cut the NDVI time series to match our y_variable dates
####################
# depends on the (str) input of which series to work with: 
#           'TPV' 'TOI' 'ROI' or 'RPV'
y = y_TPV.copy()
####################
ndvi_TS = ndvi_TS[:,:,np.where(ndvi_obsDates >= y.index.get_values()[0])[0]]
ndvi_obsDates = ndvi_obsDates[np.where(ndvi_obsDates >= y.index.get_values()[0])[0]]


# aggregate data of the pixels time series per cluster
meanTS, DBA, iterdDBA = utils_loc.gen_clusterTS(ndvi_TS, ndvi_clust)
utils_loc.gridPlotsTS(iterdDBA, cve_mpo)

# generate DataFramewitht TS of the iterdDBA of each cluster 
ndvi_df2 = utils_loc.gen_clustersDF(ndvi_obsDates, ndvi_clust, iterdDBA)


"""Esto va a ser una función dsps: """




"""Esto va a ser una función dsps: """
y = y_TPV.copy()
y.index.get_values()[0] , y.index.get_values()[-1]
y
pd.date_range(y.index.get_values()[0] -1, end_date, freq='1M')

import imp
imp.reload(utils_loc)