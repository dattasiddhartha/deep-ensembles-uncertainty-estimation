from __future__ import print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
import os
import keras
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import os
import pickle

# https://towardsdatascience.com/multi-layer-perceptron-using-tensorflow-9f3e218a4809

def MLP(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    
    print("Without Dropout, Vanilla MLP")

    import tensorflow as tf
    from sklearn.metrics import roc_auc_score, accuracy_score
    s = tf.InteractiveSession()

    ## Defining various initialization parameters for 784-512-256-10 MLP model
    num_classes = Y_train.shape[1]
    num_features = X_train.shape[1]
    num_output = Y_train.shape[1]
    num_layers_0 = 200
    num_layers_1 = 200
    num_layers_2 = 200
    starter_learning_rate = 0.001
    regularizer_rate = 0.1

    # Placeholders for the input data
    input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
    input_Y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')



    ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
    weights_0 = tf.Variable(tf.random_normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
    bias_0 = tf.Variable(tf.random_normal([num_layers_0]))
    weights_1 = tf.Variable(tf.random_normal([num_layers_0,num_layers_1], stddev=(1/tf.sqrt(float(num_layers_0)))))
    bias_1 = tf.Variable(tf.random_normal([num_layers_1]))
    weights_2 = tf.Variable(tf.random_normal([num_layers_1,num_layers_2], stddev=(1/tf.sqrt(float(num_layers_1)))))
    bias_2 = tf.Variable(tf.random_normal([num_layers_2]))
    weights_3 = tf.Variable(tf.random_normal([num_layers_2,num_output], stddev=(1/tf.sqrt(float(num_layers_2)))))
    bias_3 = tf.Variable(tf.random_normal([num_output]))

    # for dropout layer
    # keep_prob = tf.placeholder(tf.float32)
    # # Initializing weigths and biases -- with dropout
    # hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
    # hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)
    # hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0_0,weights_1)+bias_1)
    # hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)
    # hidden_output_2 = tf.nn.relu(tf.matmul(hidden_output_1_1,weights_2)+bias_2)
    # hidden_output_2_2 = tf.nn.dropout(hidden_output_2, keep_prob)
    # predicted_y = tf.sigmoid(tf.matmul(hidden_output_2_2,weights_3) + bias_3)

    # ## Initializing weigths and biases -- withOUT dropout
    hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
    hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0,weights_1)+bias_1)
    hidden_output_2 = tf.nn.relu(tf.matmul(hidden_output_1,weights_2)+bias_2)
    predicted_Y = tf.sigmoid(tf.matmul(hidden_output_2,weights_3) + bias_3)

    ## Defining the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_Y,labels=input_Y)) \
            + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)) + tf.reduce_sum(tf.square(bias_2)))

    ## Variable learning rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
    ## Adam optimzer for finding the right weight
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,weights_2,weights_3,
                                                                             bias_0,bias_1,bias_2,bias_3])

    ## Metrics definition
    correct_prediction = tf.equal(tf.argmax(Y_train,1), tf.argmax(predicted_Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ## Training parameters
    batch_size = 128
    epochs=14
    dropout_prob = 0.1
    training_accuracy = []
    training_loss = []
    testing_accuracy = []
    s.run(tf.global_variables_initializer())
    for epoch in range(epochs):    
        arr = np.arange(X_train.shape[0])
        np.random.shuffle(arr)
        for index in range(0,X_train.shape[0],batch_size):
    #         s.run(optimizer, {input_X: X_train[arr[index:index+batch_size]],
    #                           input_Y: Y_train[arr[index:index+batch_size]],
    #                         keep_prob:dropout_prob})
            s.run(optimizer, {input_X: X_train[arr[index:index+batch_size]],
                              input_Y: Y_train[arr[index:index+batch_size]]})
        training_accuracy.append(s.run(accuracy, feed_dict= {input_X:X_train, 
                                                             input_Y: Y_train}))
        training_loss.append(s.run(loss, {input_X: X_train, 
                                          input_Y: Y_train}))

    #     training_accuracy.append(s.run(accuracy, feed_dict= {input_X:X_train, 
    #                                                          input_Y: Y_train,keep_prob:1}))
    #     training_loss.append(s.run(loss, {input_X: X_train, 
    #                                       input_Y: Y_train,keep_prob:1}))

        ## Evaluation of model
    #     testing_accuracy.append(accuracy_score(Y_test.argmax(1), 
    #                             s.run(predicted_Y, {input_X: X_test,keep_prob:1}).argmax(1)))
        testing_accuracy.append(accuracy_score(Y_test.argmax(1), 
                                s.run(predicted_Y, {input_X: X_test}).argmax(1)))
        print("Epoch:{0} | Train loss: {1:.2f} | Train acc: {2:.3f} | Test acc:{3:.3f}".format(epoch,
                                                                        training_loss[epoch],
                                                                        training_accuracy[epoch],
                                                                       testing_accuracy[epoch]))
    
    # return class probabilities
#     prediction=tf.argmax(predicted_Y,1) # predicted_Y or Y_test ?
#     print("Predictions", prediction.eval(feed_dict={input_X: X_test}, session=s))
#     pred_values = prediction.eval(feed_dict={input_X: X_test}, session=s) # https://github.com/tensorflow/tensorflow/issues/97
#     prediction = tf.nn.softmax(X_test)
    prediction = s.run(predicted_Y, {input_X: X_test})
    acc = accuracy_score(Y_test.argmax(1), s.run(predicted_Y, {input_X: X_test}).argmax(1))
    
#     pred_values, 
    return prediction, acc

def MCDropout(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    
    # https://towardsdatascience.com/multi-layer-perceptron-using-tensorflow-9f3e218a4809

    print("With Dropout & Batch Normalization")

    import tensorflow as tf
    from sklearn.metrics import roc_auc_score, accuracy_score

    from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
    # Batch normalization implementation
    # from https://github.com/tensorflow/tensorflow/issues/1122
    def batch_norm_layer(inputT, is_training=True, scope=None):
        # Note: is_training is tf.placeholder(tf.bool) type
        return tf.cond(is_training,
                        lambda: batch_norm(inputT, is_training=True,
                        center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                        lambda: batch_norm(inputT, is_training=False,
                        center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                        scope=scope, reuse = True))


    s = tf.InteractiveSession()

    ## Defining various initialization parameters for 784-512-256-10 MLP model
    num_classes = Y_train.shape[1]
    num_features = X_train.shape[1]
    num_output = Y_train.shape[1]
    num_layers_0 = 200
    num_layers_1 = 200
    num_layers_2 = 200
    starter_learning_rate = 0.001
    regularizer_rate = 0.1

    # Placeholders for the input data
    input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
    input_Y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')



    ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
    weights_0 = tf.Variable(tf.random_normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
    bias_0 = tf.Variable(tf.random_normal([num_layers_0]))
    weights_1 = tf.Variable(tf.random_normal([num_layers_0,num_layers_1], stddev=(1/tf.sqrt(float(num_layers_0)))))
    bias_1 = tf.Variable(tf.random_normal([num_layers_1]))
    weights_2 = tf.Variable(tf.random_normal([num_layers_1,num_layers_2], stddev=(1/tf.sqrt(float(num_layers_1)))))
    bias_2 = tf.Variable(tf.random_normal([num_layers_2]))
    weights_3 = tf.Variable(tf.random_normal([num_layers_2,num_output], stddev=(1/tf.sqrt(float(num_layers_2)))))
    bias_3 = tf.Variable(tf.random_normal([num_output]))

    # for dropout layer
    keep_prob = tf.placeholder(tf.float32)
    # Initializing weigths and biases -- with dropout
    h0 = tf.matmul(input_X,weights_0)+bias_0
    batch_mean0, batch_var0 = tf.nn.moments(h0,[0])
    hidden_output_0 = tf.nn.relu(tf.nn.batch_normalization(h0, batch_mean0, batch_var0, tf.Variable(tf.zeros([200])), tf.Variable(tf.ones([200])), 1e-3))
    hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)
    h1 = tf.matmul(hidden_output_0,weights_1)+bias_1
    batch_mean1, batch_var1 = tf.nn.moments(h1,[0])
    hidden_output_1 = tf.nn.relu(tf.nn.batch_normalization(h1, batch_mean1, batch_var1, tf.Variable(tf.zeros([200])), tf.Variable(tf.ones([200])), 1e-3))
    hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)
    h2 = tf.matmul(hidden_output_1,weights_2)+bias_2
    batch_mean2, batch_var2 = tf.nn.moments(h2,[0])
    hidden_output_2 = tf.nn.relu(tf.nn.batch_normalization(h2, batch_mean2, batch_var2, tf.Variable(tf.zeros([200])), tf.Variable(tf.ones([200])), 1e-3))
    hidden_output_2_2 = tf.nn.dropout(hidden_output_2, keep_prob)
    predicted_Y = tf.sigmoid(tf.matmul(hidden_output_2_2,weights_3) + bias_3)



    # ## Initializing weigths and biases -- withOUT dropout
    # hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
    # hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0,weights_1)+bias_1)
    # hidden_output_2 = tf.nn.relu(tf.matmul(hidden_output_1,weights_2)+bias_2)
    # predicted_Y = tf.sigmoid(tf.matmul(hidden_output_2,weights_3) + bias_3)

    ## Defining the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_Y,labels=input_Y)) \
            + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)) + tf.reduce_sum(tf.square(bias_2)))

    ## Variable learning rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
    ## Adam optimzer for finding the right weight
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,weights_2,weights_3,
                                                                             bias_0,bias_1,bias_2,bias_3])

    ## Metrics definition
    correct_prediction = tf.equal(tf.argmax(Y_train,1), tf.argmax(predicted_Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ## Training parameters
    batch_size = 128
    epochs=14
    dropout_prob = 0.1
    training_accuracy = []
    training_loss = []
    testing_accuracy = []
    s.run(tf.global_variables_initializer())
    for epoch in range(epochs):    
        arr = np.arange(X_train.shape[0])
        np.random.shuffle(arr)
        for index in range(0,X_train.shape[0],batch_size):
            s.run(optimizer, {input_X: X_train[arr[index:index+batch_size]],
                              input_Y: Y_train[arr[index:index+batch_size]],
                            keep_prob:dropout_prob})

        training_accuracy.append(s.run(accuracy, feed_dict= {input_X:X_train, 
                                                             input_Y: Y_train,keep_prob:1}))
        training_loss.append(s.run(loss, {input_X: X_train, 
                                          input_Y: Y_train,keep_prob:1}))

        ## Evaluation of model
        testing_accuracy.append(accuracy_score(Y_test.argmax(1), 
                                s.run(predicted_Y, {input_X: X_test,keep_prob:1}).argmax(1)))
    #     testing_accuracy.append(accuracy_score(Y_test.argmax(1), 
    #                             s.run(predicted_Y, {input_X: X_test}).argmax(1)))
        print("Epoch:{0} | Train loss: {1:.2f} | Train acc: {2:.3f} | Test acc:{3:.3f}".format(epoch,
                                                                        training_loss[epoch],
                                                                        training_accuracy[epoch],
                                                                       testing_accuracy[epoch]))

    # https://r2rt.com/implementing-batch-normalization-in-tensorflow.html

#     prediction = s.run(predicted_Y, {input_X: X_test,keep_prob:dropout_prob})
    
#     return prediction

    prediction = s.run(predicted_Y, {input_X: X_test,keep_prob:1})
    acc = accuracy_score(Y_test.argmax(1), s.run(predicted_Y, {input_X: X_test,keep_prob:1}).argmax(1))

    return prediction, acc

def MLP_adversarialtraining(X_train, Y_train, X_val, Y_val, X_test, Y_test):
        ##########################################################################################################
    ################################## GRADIENTS #############################################################
    ##########################################################################################################

    # https://towardsdatascience.com/multi-layer-perceptron-using-tensorflow-9f3e218a4809

    print("Without Dropout & Batch Normalization")

    import tensorflow as tf
    from sklearn.metrics import roc_auc_score, accuracy_score

    from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
    # Batch normalization implementation
    # from https://github.com/tensorflow/tensorflow/issues/1122
    def batch_norm_layer(inputT, is_training=True, scope=None):
        # Note: is_training is tf.placeholder(tf.bool) type
        return tf.cond(is_training,
                        lambda: batch_norm(inputT, is_training=True,
                        center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                        lambda: batch_norm(inputT, is_training=False,
                        center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                        scope=scope, reuse = True))


    s = tf.InteractiveSession()

    ## Defining various initialization parameters for 784-512-256-10 MLP model
    num_classes = Y_train.shape[1]
    num_features = X_train.shape[1]
    num_output = Y_train.shape[1]
    num_layers_0 = 200
    num_layers_1 = 200
    num_layers_2 = 200
    starter_learning_rate = 0.001
    regularizer_rate = 0.1

    # Placeholders for the input data
    input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
    input_Y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')



    ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
    weights_0 = tf.Variable(tf.random_normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
    bias_0 = tf.Variable(tf.random_normal([num_layers_0]))
    weights_1 = tf.Variable(tf.random_normal([num_layers_0,num_layers_1], stddev=(1/tf.sqrt(float(num_layers_0)))))
    bias_1 = tf.Variable(tf.random_normal([num_layers_1]))
    weights_2 = tf.Variable(tf.random_normal([num_layers_1,num_layers_2], stddev=(1/tf.sqrt(float(num_layers_1)))))
    bias_2 = tf.Variable(tf.random_normal([num_layers_2]))
    weights_3 = tf.Variable(tf.random_normal([num_layers_2,num_output], stddev=(1/tf.sqrt(float(num_layers_2)))))
    bias_3 = tf.Variable(tf.random_normal([num_output]))

    # for dropout layer
    keep_prob = tf.placeholder(tf.float32)
    # Initializing weigths and biases -- with dropout
    h0 = tf.matmul(input_X,weights_0)+bias_0
    batch_mean0, batch_var0 = tf.nn.moments(h0,[0])
    hidden_output_0 = tf.nn.relu(tf.nn.batch_normalization(h0, batch_mean0, batch_var0, tf.Variable(tf.zeros([200])), tf.Variable(tf.ones([200])), 1e-3))
#     hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)
    h1 = tf.matmul(hidden_output_0,weights_1)+bias_1
    batch_mean1, batch_var1 = tf.nn.moments(h1,[0])
    hidden_output_1 = tf.nn.relu(tf.nn.batch_normalization(h1, batch_mean1, batch_var1, tf.Variable(tf.zeros([200])), tf.Variable(tf.ones([200])), 1e-3))
#     hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)
    h2 = tf.matmul(hidden_output_1,weights_2)+bias_2
    batch_mean2, batch_var2 = tf.nn.moments(h2,[0])
    hidden_output_2 = tf.nn.relu(tf.nn.batch_normalization(h2, batch_mean2, batch_var2, tf.Variable(tf.zeros([200])), tf.Variable(tf.ones([200])), 1e-3))
#     hidden_output_2_2 = tf.nn.dropout(hidden_output_2, keep_prob)
    predicted_Y = tf.sigmoid(tf.matmul(hidden_output_2,weights_3) + bias_3)



    # ## Initializing weigths and biases -- withOUT dropout
    # hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
    # hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0,weights_1)+bias_1)
    # hidden_output_2 = tf.nn.relu(tf.matmul(hidden_output_1,weights_2)+bias_2)
    # predicted_Y = tf.sigmoid(tf.matmul(hidden_output_2,weights_3) + bias_3)

    ## Defining the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_Y,labels=input_Y)) \
            + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)) + tf.reduce_sum(tf.square(bias_2)))

    ## Variable learning rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
    ## Adam optimzer for finding the right weight
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,weights_2,weights_3,
                                                                             bias_0,bias_1,bias_2,bias_3])
    grads_wrt_input_tensor = tf.gradients(loss,input_X)[0]
    # preoptimizer = tf.train.AdamOptimizer(learning_rate)
    # # grads = preoptimizer.compute_gradients(input_Y)
    # grads = preoptimizer.compute_gradients(loss)
    # grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape()), grad[1]) for grad in grads]
    # optimizer = preoptimizer.minimize(loss,var_list=[weights_0,weights_1,weights_2,weights_3,bias_0,bias_1,bias_2,bias_3])

    ## Metrics definition
    correct_prediction = tf.equal(tf.argmax(Y_train,1), tf.argmax(predicted_Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ## Training parameters
    batch_size = 128
    epochs=14
    dropout_prob = 0.1
    training_accuracy = []
    training_loss = []
    testing_accuracy = []
    s.run(tf.global_variables_initializer())
    for epoch in range(epochs):    
        arr = np.arange(X_train.shape[0])
        np.random.shuffle(arr)
        for index in range(0,X_train.shape[0],batch_size):
            _, grads_wrt_input = s.run([optimizer, grads_wrt_input_tensor], {input_X: X_train[arr[index:index+batch_size]],
                              input_Y: Y_train[arr[index:index+batch_size]]})

    #         vars_with_grads = s.run(grads, feed_dict={input_X: X_train[arr[index:index+batch_size]],
    #                           input_Y: Y_train[arr[index:index+batch_size]],
    #                         keep_prob:dropout_prob})

        training_accuracy.append(s.run(accuracy, feed_dict= {input_X:X_train, 
                                                             input_Y: Y_train}))
        training_loss.append(s.run(loss, {input_X: X_train, 
                                          input_Y: Y_train}))

        ## Evaluation of model
        testing_accuracy.append(accuracy_score(Y_test.argmax(1), 
                                s.run(predicted_Y, {input_X: X_test}).argmax(1)))
    #     testing_accuracy.append(accuracy_score(Y_test.argmax(1), 
    #                             s.run(predicted_Y, {input_X: X_test}).argmax(1)))
        print("Epoch:{0} | Train loss: {1:.2f} | Train acc: {2:.3f} | Test acc:{3:.3f}".format(epoch,
                                                                        training_loss[epoch],
                                                                        training_accuracy[epoch],
                                                                       testing_accuracy[epoch]))
    # grad_vals = s.run([grad[0] for grad in grads])
    # https://r2rt.com/implementing-batch-normalization-in-tensorflow.html

    ##########################################################################################################
    ################################## FGSM #############################################################
    ##########################################################################################################

    prop_random = 0.2 # what proportion of the training set do you want to apply perturbations to (!= sampling!!!)
    # Might not be wise to shuffle the dataset, given that we kept X and Y separate -- let's just generate random index to start from
    import random
    # last_index_to_sample_from = int(X_train.shape[0] - X_train.shape[0]*prop_random)
    # start_index = random.randint(0,last_index_to_sample_from)
    # end_index = start_index + int(X_train.shape[0]*prop_random)
    # print("Sample index range: ", (start_index, end_index))

    ind = np.arange(int(X_train.shape[0]))
    np.random.shuffle(ind)#[0:int(X_train.shape[0]*prop_random)]
    ind = ind[0:int(X_train.shape[0]*prop_random)]

    # sample from training set, matrix addition
    # sample_X_train = X_train[start_index,end_index] # X_train[[start_index,end_index]] ; this would let us take specific indices, so we can randomize the order, ensure each iteration is taking precisely unique samples
    # sample_Y_train = Y_train[start_index,end_index]

    sample_X_train = X_train[[ind]] # X_train[[start_index,end_index]] ; this would let us take specific indices, so we can randomize the order, ensure each iteration is taking precisely unique samples
    sample_Y_train = Y_train[[ind]]

    import keras.backend as K
    # from attack_utils import gen_grad

    def fgsm(x, grad, eps=0.3, clipping=True):
        """
        FGSM attack.
        """
        # signed gradient
        normed_grad = K.sign(grad).eval()

        # Multiply by constant epsilon
        scaled_grad = eps * normed_grad

        # Add perturbation to original example to obtain adversarial example
        adv_x = K.stop_gradient(x + scaled_grad)

        if clipping:
            adv_x = K.clip(adv_x, 0, 1)
        return adv_x

    # x_adv = fgsm(sample_X_train, grads_wrt_input[-1], eps=16/255, clipping=True).eval()

    ##########################################################################################################
    ################################## Adversarial Training #############################################################
    ##########################################################################################################

    # execute adversarial training --> compile into single pipeline to be turned into SOLID-able pipeline component

    # original model

    # https://towardsdatascience.com/multi-layer-perceptron-using-tensorflow-9f3e218a4809

    print("Without Dropout & Batch Normalization & Adversarial Training")

    import tensorflow as tf
    from sklearn.metrics import roc_auc_score, accuracy_score

    from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
    # Batch normalization implementation
    # from https://github.com/tensorflow/tensorflow/issues/1122
    def batch_norm_layer(inputT, is_training=True, scope=None):
        # Note: is_training is tf.placeholder(tf.bool) type
        return tf.cond(is_training,
                        lambda: batch_norm(inputT, is_training=True,
                        center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                        lambda: batch_norm(inputT, is_training=False,
                        center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                        scope=scope, reuse = True))


    s = tf.InteractiveSession()

    prop_random = 0.2 # what proportion of the training set do you want to apply perturbations to (!= sampling!!!)
    adv_placeholders = tf.zeros([int(Y_train.shape[0]*prop_random) * Y_train.shape[1]]).eval().reshape(int(Y_train.shape[0]*prop_random), Y_train.shape[1])
    modified_Y_train = np.array([list(Y_train)+list(adv_placeholders)][0])

    ## Defining various initialization parameters for 784-512-256-10 MLP model
    num_classes = Y_train.shape[1]
    num_features = X_train.shape[1]
    num_output = Y_train.shape[1]
    num_layers_0 = 200
    num_layers_1 = 200
    num_layers_2 = 200
    starter_learning_rate = 0.001
    regularizer_rate = 0.1

    # Placeholders for the input data
    input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
    input_Y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')



    ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
    weights_0 = tf.Variable(tf.random_normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
    bias_0 = tf.Variable(tf.random_normal([num_layers_0]))
    weights_1 = tf.Variable(tf.random_normal([num_layers_0,num_layers_1], stddev=(1/tf.sqrt(float(num_layers_0)))))
    bias_1 = tf.Variable(tf.random_normal([num_layers_1]))
    weights_2 = tf.Variable(tf.random_normal([num_layers_1,num_layers_2], stddev=(1/tf.sqrt(float(num_layers_1)))))
    bias_2 = tf.Variable(tf.random_normal([num_layers_2]))
    weights_3 = tf.Variable(tf.random_normal([num_layers_2,num_output], stddev=(1/tf.sqrt(float(num_layers_2)))))
    bias_3 = tf.Variable(tf.random_normal([num_output]))

    # for dropout layer
    keep_prob = tf.placeholder(tf.float32)
    # Initializing weigths and biases -- with dropout
    h0 = tf.matmul(input_X,weights_0)+bias_0
    batch_mean0, batch_var0 = tf.nn.moments(h0,[0])
    hidden_output_0 = tf.nn.relu(tf.nn.batch_normalization(h0, batch_mean0, batch_var0, tf.Variable(tf.zeros([200])), tf.Variable(tf.ones([200])), 1e-3))
#     hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)
    h1 = tf.matmul(hidden_output_0,weights_1)+bias_1
    batch_mean1, batch_var1 = tf.nn.moments(h1,[0])
    hidden_output_1 = tf.nn.relu(tf.nn.batch_normalization(h1, batch_mean1, batch_var1, tf.Variable(tf.zeros([200])), tf.Variable(tf.ones([200])), 1e-3))
#     hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)
    h2 = tf.matmul(hidden_output_1,weights_2)+bias_2
    batch_mean2, batch_var2 = tf.nn.moments(h2,[0])
    hidden_output_2 = tf.nn.relu(tf.nn.batch_normalization(h2, batch_mean2, batch_var2, tf.Variable(tf.zeros([200])), tf.Variable(tf.ones([200])), 1e-3))
#     hidden_output_2_2 = tf.nn.dropout(hidden_output_2, keep_prob)
    predicted_Y = tf.sigmoid(tf.matmul(hidden_output_2,weights_3) + bias_3)



    # ## Initializing weigths and biases -- withOUT dropout
    # hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
    # hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0,weights_1)+bias_1)
    # hidden_output_2 = tf.nn.relu(tf.matmul(hidden_output_1,weights_2)+bias_2)
    # predicted_Y = tf.sigmoid(tf.matmul(hidden_output_2,weights_3) + bias_3)

    ## Defining the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_Y,labels=input_Y)) \
            + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)) + tf.reduce_sum(tf.square(bias_2)))

    ## Variable learning rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
    ## Adam optimzer for finding the right weight
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,weights_2,weights_3,
                                                                             bias_0,bias_1,bias_2,bias_3])
    # grads_wrt_input_tensor = tf.gradients(loss,input_X)[0]
    # preoptimizer = tf.train.AdamOptimizer(learning_rate)
    # # grads = preoptimizer.compute_gradients(input_Y)
    # grads = preoptimizer.compute_gradients(loss)
    # grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape()), grad[1]) for grad in grads]
    # optimizer = preoptimizer.minimize(loss,var_list=[weights_0,weights_1,weights_2,weights_3,bias_0,bias_1,bias_2,bias_3])

    ## Metrics definition
    correct_prediction = tf.equal(tf.argmax(modified_Y_train,1), tf.argmax(predicted_Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ## Training parameters
    batch_size = 128
    epochs=14
    dropout_prob = 0.1
    training_accuracy = []
    training_loss = []
    testing_accuracy = []
    s.run(tf.global_variables_initializer())
    for epoch in range(epochs):    

        # regenerate training set
        # Might not be wise to shuffle the dataset, given that we kept X and Y separate -- let's just generate random index to start from
        import random

        ind = np.arange(int(X_train.shape[0]))
        np.random.shuffle(ind)#[0:int(X_train.shape[0]*prop_random)]
        ind = ind[0:int(X_train.shape[0]*prop_random)]

        sample_X_train = X_train[[ind]] # X_train[[start_index,end_index]] ; this would let us take specific indices, so we can randomize the order, ensure each iteration is taking precisely unique samples
        sample_Y_train = Y_train[[ind]]
        x_adv = fgsm(sample_X_train, grads_wrt_input[-1], eps=2.55/255, clipping=True).eval()
        adv_combined_X_train = np.array([list(X_train)+list(x_adv)][0])
        adv_combined_Y_train = np.array([list(Y_train)+list(sample_Y_train)][0])


        arr = np.arange(adv_combined_X_train.shape[0])
        np.random.shuffle(arr)
        for index in range(0,adv_combined_X_train.shape[0],batch_size):
            s.run(optimizer, {input_X: adv_combined_X_train[arr[index:index+batch_size]],
                              input_Y: adv_combined_Y_train[arr[index:index+batch_size]]})

    #         vars_with_grads = s.run(grads, feed_dict={input_X: X_train[arr[index:index+batch_size]],
    #                           input_Y: Y_train[arr[index:index+batch_size]],
    #                         keep_prob:dropout_prob})

        training_accuracy.append(s.run(accuracy, feed_dict= {input_X:adv_combined_X_train, 
                                                             input_Y: adv_combined_Y_train}))
        training_loss.append(s.run(loss, {input_X: adv_combined_X_train, 
                                          input_Y: adv_combined_Y_train}))

        ## Evaluation of model
        testing_accuracy.append(accuracy_score(Y_test.argmax(1), 
                                s.run(predicted_Y, {input_X: X_test}).argmax(1)))
    #     testing_accuracy.append(accuracy_score(Y_test.argmax(1), 
    #                             s.run(predicted_Y, {input_X: X_test}).argmax(1)))
        print("Epoch:{0} | Train loss: {1:.2f} | Train acc: {2:.3f} | Test acc:{3:.3f}".format(epoch,
                                                                        training_loss[epoch],
                                                                        training_accuracy[epoch],
                                                                       testing_accuracy[epoch]))
    # grad_vals = s.run([grad[0] for grad in grads])
    # https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
#     prediction = s.run(predicted_Y, {input_X: X_test})
    
#     return prediction

    prediction = s.run(predicted_Y, {input_X: X_test})
    acc = accuracy_score(Y_test.argmax(1), s.run(predicted_Y, {input_X: X_test}).argmax(1))
    
#     pred_values, 
    return prediction, acc




def MLP_randomperturbation(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    
    def random_vector_surface(shape=(32, 32, 3)):
        # generates a random vector on the surface of hypersphere
        mat = np.random.normal(size=shape)
        norm = np.linalg.norm(mat)
        return mat/norm

    def random_vector_volume(shape=(32, 32, 3)):
        # generates a random vector in the volume of unit hypersphere
        d = np.random.rand() ** (1 / np.prod(shape))

        return random_vector_surface() * d

    print("Without Dropout, MLP with random perturbations")

    prop_random = 0.2 # what proportion of the training set do you want to apply perturbations to (!= sampling!!!)
    # random_vector_surface()

    vec1 = random_vector_surface(shape=(int(X_train.shape[0]*prop_random), X_train.shape[1]))
    vec2 = random_vector_volume(shape=(int(X_train.shape[0]*prop_random), X_train.shape[1]))
    pert = np.zeros((int(X_train.shape[0]*prop_random), X_train.shape[1]), dtype=np.float32)

    # pert = (eps/255.0) * np.sign(vec1) + (rad/255.0) * vec2 # use of epsilon=16 for reference


    # Might not be wise to shuffle the dataset, given that we kept X and Y separate -- let's just generate random index to start from
    import random
    # last_index_to_sample_from = int(X_train.shape[0] - X_train.shape[0]*prop_random)
    # start_index = random.randint(0,last_index_to_sample_from)
    # end_index = start_index + int(X_train.shape[0]*prop_random)
    # print("Sample index range: ", (start_index, end_index))

    ind = np.arange(int(X_train.shape[0]))
    np.random.shuffle(ind)#[0:int(X_train.shape[0]*prop_random)]
    ind = ind[0:int(X_train.shape[0]*prop_random)]

    # sample from training set, matrix addition
    # sample_X_train = X_train[start_index,end_index] # X_train[[start_index,end_index]] ; this would let us take specific indices, so we can randomize the order, ensure each iteration is taking precisely unique samples
    # sample_Y_train = Y_train[start_index,end_index]

    sample_X_train = X_train[[ind]] # X_train[[start_index,end_index]] ; this would let us take specific indices, so we can randomize the order, ensure each iteration is taking precisely unique samples
    sample_Y_train = Y_train[[ind]]

    import tensorflow as tf
    sess = tf.InteractiveSession()

    # Some tensor we want to print the value of
    random_perturbed_X_train = tf.add(sample_X_train,vec1)
    # warning: no perturbation for labels, ground truth

    # Add print operation
    random_perturbed_X_train_print = tf.Print(random_perturbed_X_train, [random_perturbed_X_train], message="This is a: ")

    # Add more elements of the graph using a
    # b = tf.add(a, a)

    random_perturbed_X_train_array = random_perturbed_X_train_print.eval().copy()

    # combine two arrays into master array, combine two label arrays into master array

    random_combined_X_train = np.array([list(X_train)+list(random_perturbed_X_train_array)][0])
    random_combined_Y_train = np.array([list(Y_train)+list(sample_Y_train)][0])
    print("New training set shape (X,Y): ", (random_combined_X_train.shape, random_combined_Y_train.shape))


    import tensorflow as tf
    from sklearn.metrics import roc_auc_score, accuracy_score
    s = tf.InteractiveSession()

    ## Defining various initialization parameters for 784-512-256-10 MLP model
    num_classes = random_combined_Y_train.shape[1]
    num_features = random_combined_X_train.shape[1]
    num_output = random_combined_Y_train.shape[1]
    num_layers_0 = 200
    num_layers_1 = 200
    num_layers_2 = 200
    starter_learning_rate = 0.001
    regularizer_rate = 0.1

    # Placeholders for the input data
    input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
    input_Y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')



    ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
    weights_0 = tf.Variable(tf.random_normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
    bias_0 = tf.Variable(tf.random_normal([num_layers_0]))
    weights_1 = tf.Variable(tf.random_normal([num_layers_0,num_layers_1], stddev=(1/tf.sqrt(float(num_layers_0)))))
    bias_1 = tf.Variable(tf.random_normal([num_layers_1]))
    weights_2 = tf.Variable(tf.random_normal([num_layers_1,num_layers_2], stddev=(1/tf.sqrt(float(num_layers_1)))))
    bias_2 = tf.Variable(tf.random_normal([num_layers_2]))
    weights_3 = tf.Variable(tf.random_normal([num_layers_2,num_output], stddev=(1/tf.sqrt(float(num_layers_2)))))
    bias_3 = tf.Variable(tf.random_normal([num_output]))

    # for dropout layer
    # keep_prob = tf.placeholder(tf.float32)
    # # Initializing weigths and biases -- with dropout
    # hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
    # hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)
    # hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0_0,weights_1)+bias_1)
    # hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)
    # hidden_output_2 = tf.nn.relu(tf.matmul(hidden_output_1_1,weights_2)+bias_2)
    # hidden_output_2_2 = tf.nn.dropout(hidden_output_2, keep_prob)
    # predicted_y = tf.sigmoid(tf.matmul(hidden_output_2_2,weights_3) + bias_3)

    # ## Initializing weigths and biases -- withOUT dropout
    hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
    hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0,weights_1)+bias_1)
    hidden_output_2 = tf.nn.relu(tf.matmul(hidden_output_1,weights_2)+bias_2)
    predicted_Y = tf.sigmoid(tf.matmul(hidden_output_2,weights_3) + bias_3)

    ## Defining the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_Y,labels=input_Y)) \
            + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)) + tf.reduce_sum(tf.square(bias_2)))

    ## Variable learning rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
    ## Adam optimzer for finding the right weight
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,weights_2,weights_3,
                                                                             bias_0,bias_1,bias_2,bias_3])

    ## Metrics definition
    correct_prediction = tf.equal(tf.argmax(random_combined_Y_train,1), tf.argmax(predicted_Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ## Training parameters
    batch_size = 128
    epochs=14
    dropout_prob = 0.1
    training_accuracy = []
    training_loss = []
    testing_accuracy = []
    s.run(tf.global_variables_initializer())
    for epoch in range(epochs):    
        arr = np.arange(X_train.shape[0])
        np.random.shuffle(arr)
        for index in range(0,random_combined_X_train.shape[0],batch_size):
    #         s.run(optimizer, {input_X: X_train[arr[index:index+batch_size]],
    #                           input_Y: Y_train[arr[index:index+batch_size]],
    #                         keep_prob:dropout_prob})
            s.run(optimizer, {input_X: random_combined_X_train[arr[index:index+batch_size]],
                              input_Y: random_combined_Y_train[arr[index:index+batch_size]]})
        training_accuracy.append(s.run(accuracy, feed_dict= {input_X:random_combined_X_train, 
                                                             input_Y: random_combined_Y_train}))
        training_loss.append(s.run(loss, {input_X: random_combined_X_train, 
                                          input_Y: random_combined_Y_train}))

    #     training_accuracy.append(s.run(accuracy, feed_dict= {input_X:X_train, 
    #                                                          input_Y: Y_train,keep_prob:1}))
    #     training_loss.append(s.run(loss, {input_X: X_train, 
    #                                       input_Y: Y_train,keep_prob:1}))

        ## Evaluation of model
    #     testing_accuracy.append(accuracy_score(Y_test.argmax(1), 
    #                             s.run(predicted_Y, {input_X: X_test,keep_prob:1}).argmax(1)))
        testing_accuracy.append(accuracy_score(Y_test.argmax(1), 
                                s.run(predicted_Y, {input_X: X_test}).argmax(1)))
        print("Epoch:{0} | Train loss: {1:.2f} | Train acc: {2:.3f} | Test acc:{3:.3f}".format(epoch,
                                                                        training_loss[epoch],
                                                                        training_accuracy[epoch],
                                                                       testing_accuracy[epoch]))

    # return class probabilities
    #     prediction=tf.argmax(predicted_Y,1) # predicted_Y or Y_test ?
    #     print("Predictions", prediction.eval(feed_dict={input_X: X_test}, session=s))
    #     pred_values = prediction.eval(feed_dict={input_X: X_test}, session=s) # https://github.com/tensorflow/tensorflow/issues/97
    #     prediction = tf.nn.softmax(X_test)
#     prediction = s.run(predicted_Y, {input_X: X_test})
    
#     return prediction

    prediction = s.run(predicted_Y, {input_X: X_test})
    acc = accuracy_score(Y_test.argmax(1), s.run(predicted_Y, {input_X: X_test}).argmax(1))
    
#     pred_values, 
    return prediction, acc



def NLL(pred_values, MNIST_Y_test, index_of_y):

    from keras import backend as K

    # generate NLL distribution
#     pred_hotcoded = np_utils.to_categorical(pred_values, 10)[index_of_y:index_of_y+1]

    # y_test = y_test.astype('float32') # necessary here, since y_pred comes in this type - check in your case with y_test.dtype and y_pred.dtype
    # y_test = K.constant(y_test)
    # y_pred = K.constant(y_pred)

#     y_pred = K.constant(pred_hotcoded)
    
    y_pred = K.constant(pred_values[index_of_y:index_of_y+1])

    g = K.categorical_crossentropy(target=MNIST_Y_test[index_of_y:index_of_y+1], output=y_pred)  # tensor
    ce = K.eval(g)  # 'ce' for cross-entropy
    ce.shape
    # (10000,) # i.e. one loss quantity per sample

    # sum up and divide with the no. of samples:
    log_loss = np.sum(ce)/ce.shape[0]
#     log_loss
    # 0.05165323486328125
    
    # https://stackoverflow.com/questions/52497625/how-to-calculate-negative-log-likelihoog-on-mnist-dataset
    return log_loss

def entropy_values(MNIST_Y_test, prediction):
    entropy_values=[]
    for i in range(len(MNIST_Y_test)): # WARNING: Remove 100 limit, let whole dataframe!!!
        log_loss = NLL(prediction, MNIST_Y_test, i)
        entropy_values.append(log_loss)
    #     print(log_loss)
    return entropy_values