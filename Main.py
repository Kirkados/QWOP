"""
Main script that runs the D4PG learning algorithm 
(https://arxiv.org/pdf/1804.08617)

It features the standard DDPG algorithm with a number 
of improvements from other researchers.
Namely:
    Distributed rollouts           (https://arxiv.org/pdf/1602.01783)
    A distributional critic        (http://arxiv.org/abs/1707.06887)
    N-step returns                 (https://arxiv.org/pdf/1602.01783)
    Prioritized experience replay  (http://arxiv.org/abs/1511.05952)

This implementation does not use the 
ApeX framework (https://arxiv.org/abs/1803.00933) as the original authors did.
Instead, it uses the python 'threading' library.


@author: Kirk

Code started: October 15, 2018
"""

# Importing libraries & other classes

import threading
import tensorflow as tf

from agent import Agent
from learner import QNetwork
from replay_buffer import ReplayBuffer

import saver
import displayer

from settings import Settings

#%%
##########################
##### SETTING UP RUN #####
##########################

tf.reset_default_graph() # clearing tensorflow graph

# Starting tensorflow session
with tf.Session() as sess:
    
    ##############################
    ##### Initializing items #####
    ##############################
    
    saver = Saver.Saver(sess) # initializing saver class (for loading & saving data)
    displayer = Displayer.Displayer() # initializing displayer class (for displaying plots)
    replay_buffer = ReplayBuffer(Settings.PRIORITIZED_REPLAY_BUFFER) # initializing replay buffer
    
    threads = []
    
    # Placing each actor into its own thread
    for i in range(Settings.NUMBER_OF_ACTORS):
        actor = Agent(sess, i+1, displayer, replay_buffer)
        threads.append(threading.Thread(target = actor.run))
    
    # Generating the critic (which is a Q-network) and assigning it to a thread
    # May need to assign the critic to the GPU via -> with tf.device('/device:GPU:0'):
    critic = QNetwork(sess, saver, replay_buffer)
    threads.append(threading.Thread(target = critic.run))
    
    # Loaging in previous training results, if available. 
    # Otherwise, initialize all tensorflow variables.
    if not saver.load():
        sess.run(tf.global_variables_initializer())
    
    #############################################
    ##### STARTING EXECUTION OF ALL THREADS #####
    #############################################
        
    # Starting All Threads that have been appended to 'threads'
    for t in threads:
        t.start()
        
    print('Done starting!')
    
    ####################################################
    ##### Waiting until all threads have completed #####
    ####################################################
    
    print("Waiting until threads finish")
    for t in threads:
        t.join()
        
    print('Threads are done running!')
    
    # Save the final parameters
    saver.save(critic.total_training_iterations)
    