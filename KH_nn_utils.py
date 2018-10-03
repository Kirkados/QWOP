"""
Kirk Hovell's Neural Network Starter Code, built on TensorFlow
Intended to have all the tools available to build any neural network you're
currently capable of building.

Build your own graph, then move on to the training section to train it up.

Test data is provided by the MNIST dataset

Created: December 9, 2017
@author: Kirk Hovell
"""

import tensorflow as tf
#import time
#import datetime

#tf.reset_default_graph() # Wipes tensorflow graph clean

#%% This section defines functions that we'll use later on

# Generating Fully Connected Layer
def fc_layer(input,input_size,output_size,name, nonlinear, batch_norm = False):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([input_size,output_size], stddev = 0.001, name = "Initially_Random"), name = "W") # Creating weight Variable
        b = tf.Variable(tf.zeros(output_size), name = "b") # Creating bias variable
        linear = tf.matmul(input, W) + b # performing Z = W'*X + b
        if batch_norm:
            linear = tf.layers.batch_normalization(linear)
        if nonlinear == 'relu':
            activation = tf.nn.relu(linear) # Using a ReLU activation function
        elif nonlinear == 'tanh':
            activation = tf.nn.tanh(linear)
        elif nonlinear == 'sigmoid':
            activation = tf.nn.sigmoid(linear)
        
        #tf.summary.histogram("Weights", W)
        #tf.summary.histogram("Biases", b)
        #tf.summary.histogram("Activations", activation)
        return linear, activation

# Generating a Convolutional Layer
def conv_layer(input, filter_size, channels_in, channels_out, stride, pad_type, name):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([filter_size,filter_size,channels_in,channels_out], stddev =0.1, name = "Initially_Random"), name = "W")
        b = tf.Variable(tf.zeros(channels_out), name = "b")
        linear = tf.nn.conv2d(input, W, strides = [1, stride, stride, 1], padding = pad_type)
        activation = tf.nn.relu(linear)
        
        tf.summary.histogram("Weights", W)
        tf.summary.histogram("Biases", b)
        tf.summary.histogram("Activations", activation)

        return linear, activation

# Generating a maxpool layer
def maxpool_layer(input, stride, pad_type, name):
    with tf.name_scope(name):
        activation = tf.nn.max_pool(input, ksize=[1,stride,stride,1], strides=[1,stride,stride,1], padding = pad_type)
        
        return activation
        
# Generating a dropout layer
def dropout_layer(input, keep_prob):
    with tf.name_scope("Dropout"):
        dropped_out = tf.nn.dropout(input, keep_prob)
        return dropped_out

#
        ##### SAMPLE USAGE BELOW #####
##%% Generating the TensorFlow graph
#
## Importing MNIST Data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
## Creating placeholders for inputs (x) and labels (y)
#x = tf.placeholder(tf.float32, shape = [None, 784], name ="x") # Inputs
#y = tf.placeholder(tf.float32, shape = [None, 10], name = "labels") # Labels
#
#
### Building Graph
## Reshaping the images from vectors into images
#x_image = tf.reshape(x, [-1,28,28,1]) # [examples, X, Y, channels]
#
## Convolutional layer (5x5 filter, 32 channels)
#linear1, activation1 = conv_layer(x_image,5,1,32,1,"SAME",name = "Conv1") # input, filter_size, channels_in, channels_out, stride, pad_type, name
## Pooling layer, stride = 2
#pool1 = maxpool_layer(activation1,2,"SAME","pool1") # input, stride, pad_type, name
#
## Convolutional layer (5x5 filter, 64 channels)
#linear2, activation2 = conv_layer(pool1,5,32,64,1,"SAME",name = "Conv2")
#pool2 = maxpool_layer(activation2,2,"SAME","pool2") # pooling layer
#
#flat = tf.reshape(pool2,[-1,7*7*64]) # reshaping for FC layers
#
## FC layer 1, 1024 neurons
#linear3, activation3 = fc_layer(flat,7*7*64,1024, name = "FC1")
#keep_prob = tf.placeholder(tf.float32)
#dropout = dropout_layer(activation3, keep_prob) # dropout
#
## FC layer 2, 10 neurons
#linear4, activation4 = fc_layer(dropout,1024,10, name = "FC2")
#
#
##linear_1, act_1 = fc_layer(x, 784, 1000, name = "FC1")
#
## Another layer (10 neurons)
##linear_2, act_2 = fc_layer(act_1,1000,300, name = "FC2")
#
## Another layer
##linear_3, act_3 = fc_layer(act_2,300,10, name = "FC3")
#
## Choose the final layer's logits (i.e., non-relu'd), for use in the softmax
#final_logits = linear4
#
#
### Creating Cost Function
#with tf.name_scope("Cost"):
#    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = final_logits))
#    tf.summary.scalar("Cost", cost) ## WHAT DOES THIS DO
#
#
##%% Getting Training Initialized
#
#### Defining Hyperparameters
#learning_rate = 1e-4
#iterations = 2
#batch_size = 50
#output_every_num_of_iterations = 100
#save_trained_network = False
#log_tensorboard_during_training = True # Setting to Falso gives a small performance improvement
#run_name = "TestDesktop"
#
#### Choosing Optimizer
#optimizer = tf.train.AdamOptimizer(learning_rate)
#train_one_step = optimizer.minimize(cost) # using the Optimizer chosen above, adjust the Variables to reduce cost
#init = tf.global_variables_initializer() 
#saver = tf.train.Saver() # Getting ready to save
#
## Measuring Success
#with tf.name_scope("Accuracy"):
#    correct_predictions = tf.cast(tf.equal(tf.argmax(final_logits,1), tf.argmax(y,1)), tf.float32)
#    accuracy = tf.reduce_mean(correct_predictions)
#    tf.summary.scalar("Accuracy", accuracy)
#
#summ = tf.summary.merge_all()
#writer = tf.summary.FileWriter('TensorBoard/' + run_name + '-{:%Y-%m-%d %H-%M}'.format(datetime.datetime.now()))
#
#start_time = time.time()
#with tf.Session() as sess:
#    sess.run(init) # initialize the variables
#    writer.add_graph(sess.graph)
#    for i in range(iterations):
#        x_train_batch, y_train_batch = mnist.train.next_batch(batch_size)
#        
#        if i % output_every_num_of_iterations == 0:            
#            if log_tensorboard_during_training:
#                training_accuracy, s = sess.run([accuracy, summ], feed_dict={x: x_train_batch, y: y_train_batch, keep_prob:1.0})
#                writer.add_summary(s,i)
#            else:
#                training_accuracy = sess.run(accuracy, feed_dict={x: x_train_batch, y: y_train_batch, keep_prob:1.0})
#                    
#            print("Step: {}, Training accuracy: {:1.3f}" .format(i, training_accuracy))
#        
#        sess.run(train_one_step, feed_dict={x: x_train_batch, y: y_train_batch,keep_prob:0.5})
#        ## Possibly log training performance every # iterations
#            
#    training_time = time.time() - start_time # total training time
#    print('It took %g seconds to train!' %training_time)
#    
#    # Evaluating on Test Data
#    n_test_batches = mnist.test.images.shape[0] // batch_size
#    cumulative_accuracy = 0.0
#    for i in range(n_test_batches):
#        x_test_batch, y_test_batch = mnist.test.next_batch(batch_size)
#        cumulative_accuracy += sess.run(accuracy,{x: x_test_batch, y: y_test_batch,keep_prob:1.0})
#    print("Test accuracy: {:1.4f}" .format(cumulative_accuracy/n_test_batches))
#    
#    # Saving Trained Network
#    if save_trained_network:
#        save_path = saver.save(sess, 'TrainedParams/' + run_name + '-{:%Y-%m-%d %H-%M}'.format(datetime.datetime.now()) + '.ckpt')
#        print('Trained network is saved in file: %s' %save_path)
#        
## To restore variables. Note: variables do not have to be initialized beforehand
## saver = tf.train.Saver()
## saver.restore(sess, os.path.splitext(os.path.basename(__file__))[0] + '_data/saved_params.ckpt')
#
