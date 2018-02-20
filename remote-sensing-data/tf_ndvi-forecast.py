# -*- coding: utf-8 -*-
"""
Script for forecasting the NDVI observations using a Long Short Term Memory (LSTM) NN  
"""

import numpy as np
import pandas as pd
import math
import tensorflow as tf
import utils_loc


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