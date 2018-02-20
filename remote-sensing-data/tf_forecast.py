# -*- coding: utf-8 -*-
"""
Script for forecasting the NDVI observations using a Long Short Term Memory (LSTM) NN  
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import math
import itertools
import tensorflow as tf
import os
##############
os.chdir("C:\\Users\\Raul Vazquez\\Desktop\\NNproj\\remote-sensing\\remote-sensing-data")
%matplotlib auto
##############
import utils_loc
#from tf_lstm import lstm_model
from tf_cnn import cnn_model_fn
#import glob

# import NDVI data
cve_mpo, ndvi_TS, ndvi_clust, init_date, end_date = utils_loc.read_NDVI(mun_num=7, clust_cut_val=2)
init_date = np.datetime64(init_date)
end_date = np.datetime64(end_date)

y_TPV, y_RPV, y_TOI, y_ROI = utils_loc.read_agricolaDB(cve_mpo, init_date, end_date)

'''
MODIS releases images every 16 days
    1 year contains 23 observations s.t. every year the observations are on the same date 
    (the last observation of the year contains less days)
'''

YY = pd.DatetimeIndex([init_date.astype('str')]).year[0]
ndvi_obsDates =[np.arange(np.datetime64(str(YY+aa)), np.datetime64(str(YY+aa)) +  np.timedelta64(357,'D'), np.timedelta64(16,'D')) for aa in range(0,math.ceil( (end_date-init_date)/np.timedelta64(365,'D') ))]
ndvi_obsDates = np.array(ndvi_obsDates).flatten()

''' ------- PREPROCESS X variable ---------- '''

####################
# def preprocessX(y,):
# depends on which series to work with: 
#           'TPV' 'TOI' 'ROI' or 'RPV'
y = y_TPV.copy()
####################
''' # cut the NDVI time series to match our y_variable dates
ndvi_obsDates = ndvi_obsDates[np.where(ndvi_obsDates >= y.index.get_values()[0])[0]]
ndvi_obsDates = ndvi_obsDates[np.where(ndvi_obsDates <  y.index.get_values()[-1])[0]]
ndvi_TS = ndvi_TS[:,:,np.where(ndvi_obsDates >= y.index.get_values()[0])[0]]
ndvi_TS = ndvi_TS[:,:,np.where(ndvi_obsDates < y.index.get_values()[-1])[0]]
'''

#ndvi_TS2 = [ndvi_TS[:,:,i].flatten() for i in range(len(ndvi_obsDates))]
#ndvi_TS2 = [list(itertools.chain.from_iterable(ndvi_TS2[i:i+8])) for i in range(len(ndvi_TS2))]
'''
train_data = np.array()
for date in y.index.get_values()[9:10]:
    print(date)
    print(ndvi_obsDates[np.where(ndvi_obsDates <= date + np.timedelta64(30,'D'))[0][-8:]])
    [ndvi_TS[:,:,np.where(ndvi_obsDates <= date + np.timedelta64(30,'D'))[0][-8:]].flatten()
    print(save)
'''
    
"TO DO: Complete the Dates for agro_production (the y.index.get_values() series) and change [2:] to [3:]"
train_data = np.array([ndvi_TS[:,:,np.where(ndvi_obsDates <= date + np.timedelta64(30,'D'))[0][-8:]].flatten() for date in y.index.get_values()[2:]])
train_dates = np.array([ndvi_obsDates[np.where(ndvi_obsDates <= date + np.timedelta64(30,'D'))[0][-8:]].flatten() for date in y.index.get_values()[2:]])    
train_labels = np.array(pd.cut(y['prop_cosecha'], bins = [-0.01,0.25,0.5,0.75,1.0], labels = [0,1,2,3]))[2:]
"""
 LABELS = 0: 0 < proporcion_cosechada <= 0.25,
          1: 0.25< prop_cosechada <= 0.5,
          2: 0.5 < prop_cosechada <= 0.75
          3: 0.75< prop_cosechada <= 1.0
"""   

from tf_cnn import cnn_model_fn
# Create the Estimator
agrProd_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/ndvi_cnn_model")

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data.astype('float32')},
    y=train_labels.astype('int32'),
    batch_size=100,
    num_epochs=None,
    shuffle=True)
agrProd_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])

# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
eval_results = agrProd_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)


'''
# VISUALIZATION METHODS
# the y_series:
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
# visualize the TS
utils_loc.gridPlotsTS(ndvi_TS, cve_mpo)

'''



































""" DEPRECATED: do NOT run this next part!!!!!! """
# aggregate data of the pixels time series per cluster
meanTS, DBA, iterdDBA = utils_loc.gen_clusterTS(ndvi_TS, ndvi_clust)
utils_loc.gridPlotsTS(iterdDBA, cve_mpo)
# generate DataFramewitht TS of the iterdDBA of each cluster 
ndvi_df = utils_loc.gen_clustersDF(ndvi_obsDates, ndvi_clust, iterdDBA)

"""TO DO. corregir desfase... meter 4 meses antes de X"""

# plot NDVI and y variable together; helps see the relation between them
plt.figure()
plt.plot(y_TPV*4000, label = "TPV"), plt.plot(y_RPV*4000, label = "RPV"), plt.plot(ndvi_df[['c1','c2','c3']])
plt.legend()



"""Esto va a ser una función dsps: """
TIMESTEPS = 4
XX,yy = load_csvdata(y, time_steps = TIMESTEPS)


# specs
LOG_DIR = './ops_logs/lstm_weather'
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 1000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

from tensorflow.contrib import learn
regressor = learn.SKCompat(learn.Estimator(
        model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
        model_dir=LOG_DIR
))

"""IMPORTANT: Monitors are deprecated  .... change this """
validation_monitor = learn.monitors.ValidationMonitor(XX['val'], yy['val'],
                                                     every_n_steps=PRINT_STEPS,
                                                     early_stopping_rounds=1000)


regressor.fit(XX['train'], yy['train'],
              monitors=[validation_monitor],
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)

predicted = regressor.predict(XX['test'])
rmse = np.sqrt(((predicted - yy['test']) ** 2).mean(axis=0))


import imp
imp.reload(tf_cnn)




def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observations, suitable for RNNs.
    recall that RNNs observe x_1,...,x_n and try to guess x_{n+1}
    INPUT:
        - data: (pd.DataFrame) 
        - time_steps: (int) number of previous steps to take into account at every time step
        - labels: (boolean) True for y variable (objective var) and False for X variable (observed var)
    OUTPUT:
        - (np.array) 
            if labels == True: inputed data without the first time_steps observations
            if labels == False: array of size 'len(data) - time_steps'. Each entry contains an array
                                with the observation and its previous time_steps observations
    * example:
        l = pd.DataFrame([1, 2, 3, 4, 5])
        rnn_data(l , 2) = [[1, 2], [2, 3], [3, 4]]
        rnn_data(l , 2, True) =  [3, 4, 5]    
        rnn_data(l , 3) = [[1, 2, 3], [2, 3, 4]]
        rnn_data(l , 3, True) =  [4, 5]  
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)


def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    INPUT:
        - data: (pd.DataFrame)
        - time_steps: (int) number of previous steps to take into account at every time step
        - labels: (boolean) True for y variable and False for X variable
        - val_size: (double) size of the validation set (proportion, between 0 and 1)
        - test_size: (double) size of the test set
    OUTPUT:
        - train, test and validation sets of sizes:
            train = data from timestep 0 to timestep len(data) - ( len(validation) + len(test) )
            val = data from len(train) to len(data) - len(test)
            test = last observations of the time series
          The sets are returnes in a suitable structure for a RNN (*refer to rnn_data)
    """
    # compute the size of the test and validation sets
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))
    
    # divide the data depending on the previous values
    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]
    
    #df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))


def load_csvdata(rawdata, time_steps):
    '''
    Function to load the given CSV data
    INPUT:
         - rawdata: (DataFrame) the original data 
         - time_steps: (int) number of steps to consider 
    OUTPUT:
        - X: (dict) containing the train, test and validation sets with observed variable
        - y: (dict)  "             "                "                "  objective variable
    '''
    data = rawdata
    # make sure we have a pandas Data Frame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # separate into train, test and validation sets
    train_x, val_x, test_x = prepare_data(data, time_steps)
    train_y, val_y, test_y = prepare_data(data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)











