# -*- coding: utf-8 -*-
"""
Script originally created for classifying the MNIST example problem (Modified National Institute of Standards and Technology database)
"""
import tensorflow as tf

def cnn_model_fn(features, labels, mode):
    """
    Creates a deep NN model with:
        - 2 convolutional layers
        - 2 max-pooling layers
        - 2 dense layers
    INPUT:
        - features:
        - labels: 
        - mode: indicates what will be done with the model. Can be:
            tf.estimator.ModeKeys.PREDICT
            tf.estimator.ModeKeys.TRAIN
            
    OUTPUT:
        
  
    """
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) # [batch_size, image_width, image_height, channels]. 
                                                          # batch_size = -1 specifies that this dimension will be computed based on the specs of features["x"]. All other dims are kept constant.
                                                          # channels = 1 for B&W images, 3 for RGB
                                                          
    # first convolutional layer
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32,# the dim of the output space (i.e. the number of filters in the convolution)
                             kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # first pooling layer
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)# output tensor size=[batch_size, 14, 14, 32] because the 2x2 filter reduces width and height by 50% each

    # second convolutional layer
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], 
                             padding="same", activation=tf.nn.relu) # padding=same: the output tensor has the same width and height as the input tensor
  
    # second pooling layer
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)# output tensor size=[batch_size, 7, 7, 64]

    # first dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # classified OUTPUTS layer
    logits = tf.layers.dense(inputs=dropout, units=10) # Logit is a fn that maps probas to R, i.e.,[0, 1] -> (-inf, inf)
                                                    # has shape [batch_size, 10].
    
    # Generate predictions
    predictions = {
        "classes": tf.argmax(input=logits, axis=1), # predicted class
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")# probs for each possible class
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # compute the loss (this is valid for both modes, .TRAIN and .EVAL)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Training Operation (only for .TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (only for .EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)