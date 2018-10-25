"""
This script builds the QNetwork (critic in the case of D4PG) and generates
all the functions needed to train it. It also creates the agent who is 
continually trained. Target networks are used as well.

Training:
    The critic is trained using supervised learning to minimize the 
    cross-entropy loss between the Q value and the target value 
    y = r_t + gamma * Q_target(next_state, Action(next_state))
        
    To train the actor, we apply the policy gradient
    Grad = grad(Q(s,a), A)) * grad(A, params)
        
        
"""

import tensorflow as tf
import numpy as np
import time

from Model import build_actor_network, build_Q_network
from network_utilities import get_variables, copy_variables, l2_regularization

from Settings import Settings

class QNetwork:
    
    def __init__(self, sess, saver, replay_buffer):        
        # When initialized, create the trainable actor, the critic, and their respective target networks
        
        print('Q-Network initializing...')
        
        self.sess = sess
        self.saver = saver
        self.replay_buffer = replay_buffer
        
        # Creating placeholders for training data
        self.state_placeholder      = tf.placeholder(dtype = tf.float32, shape = [None, *Settings.STATE_SIZE], name = 'state_placeholder') # the '*' unpacks the STATE_SIZE (incase it's pixels of higher dimension)
        self.action_placeholder     = tf.placeholder(dtype = tf.float32, shape = [None, Settings.ACTION_SIZE], name = 'action_placeholder')
        self.reward_placeholder     = tf.placeholder(dtype = tf.float32, shape = [None], name = 'reward_placeholder')
        self.next_state_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, *Settings.STATE_SIZE], name = 'next_state_placeholder')
        self.not_done_placeholder   = tf.placeholder(dtype = tf.float32, shape = [None], name = 'not_done_placeholder')
        
        # Changing to column vectors
        self.reward_placeholder   = tf.expand_dims(self.reward_placeholder, 1)
        self.not_done_placeholder = tf.expand_dims(self.not_done_placeholder, 1)
        
        # Getting the batch size
        self.batch_size           = tf.shape(self.reward_placeholder)[0]
        
        # Setting up distributional critic items
        self.delta_z = (Settings.MAX_Q - Settings.MIN_Q) / (Settings.NUMBER_OF_ATOMS - 1)
        self.z       = tf.range(Settings.MIN_Q, Settings.MAX_Q + self.delta_z, self.delta_z)
        
        ####################################################
        #### Build the networks and training operations ####
        ####################################################
        self.build_models()
        self.build_targets()
        self.build_network_training_operations()
        self.build_target_parameters_update_operations()
        
        print('Q-Network created!')
        
    def build_models(self):
        # Build the critic and the trained actor
        
        ###############################
        ### Build the learned actor ###
        ###############################
        self.actor = build_actor_network(state = self.state_placeholder, trainable = True, scope = 'learned_actor')
        
        ################################
        ### Build the learned critic ###
        ################################
        self.q_network = build_Q_network(state = self.state_placeholder, 
                                              action = self.action_placeholder, 
                                              trainable = True, 
                                              reuse = False, 
                                              scope = 'learned_critic')
        
        #####################################################################################
        ### Build another form of the critic, where the actions are provided by the actor ###
        #####################################################################################
        self.q_network_distribution_with_actor_for_actions = build_Q_network(state = self.state_placeholder,
                                                                              action = self.actor,
                                                                              trainable = True,
                                                                              reuse = True, # this means this network takes parameters from the main self.q_network
                                                                              scope = 'learned_critic')
        
        # Turn the Q-Network distribution into a Q-value. 
        self.q_network_value_with_actor_for_action = tf.reduce_sum(self.z * self.q_network_distribution_with_actor_for_actions, axis = 1)

        
    def build_targets(self):
        # Build the target networks
        
        # Target Actor
        self.target_actor = build_actor_network(self.next_state_placeholder,
                                                trainable = False,
                                                scope = 'target_actor')
        
        # Target Q Network
        self.Q_distribution_target = build_Q_network(state = self.next_state_placeholder,
                                                          action = self.action_placeholder,
                                                          trainable = False,
                                                          reuse = False,
                                                          scope = 'target_critic')
        
    def build_target_parameters_update_operations(self):
        # updates the slowly-changing target networks according to tau
        
        # Grab parameters from the main networks
        self.actor_parameters = get_variables('learned_actor', trainable = True)
        self.critic_parameters = get_variables('learned_critic', trainable = True)
        
        self.parameters = self.actor_parameters + self.critic_parameters
        
        # Grab parameters from target networks
        self.target_actor_parameters = get_variables('target_actor', trainable = False)
        self.target_critic_parameters = get_variables('target_critic', trainable = False)
        
        self.target_parameters = self.target_actor_parameters + self.target_critic_parameters
        
        # Operation to initialize the target networks
        self.initialize_target_network_parameters = copy_variables(source_variables = self.parameters, destination_variables = self.target_parameters, tau = 1)
        
        # Updating target networks at rate tau 
        self.update_target_network_parameters     = copy_variables(source_variables = self.parameters, destination_variables = self.target_parameters, tau = Settings.TARGET_NETWORK_TAU)
        
    def build_network_training_operations(self):
        # builds the operations that are used to train the real actor and critic
        
        # distributional critic operations
        zz = tf.tile(self.z[None], [self.batch_size, 1])
        
        # Compute projection of Tz onto the support z
        Tz = tf.clip_by_value(self.reward + (Settings.DISCOUNT_FACTOR ** Settings.N_STEP_RETURN) * self.not_done * zz,
                              Settings.MIN_Q, Settings.MAX_Q - 1e-4)
        
        bj = (Tz - Settings.MIN_Q) / self.delta_z
        l = tf.floor(bj)
        u = l + 1
        l_ind, u_ind = tf.to_int32(l), tf.to_int32(u)
        
        # Initialize the critic loss
        critic_loss = tf.zeros([self.batch_size])
        
        for j in range(Settings.NUMBER_OF_ATOMS):
            # Select the value of Q(s_t, a_t) onto the atoms l and u and clip it
            l_index = tf.stack((tf.range(self.batch_size), l_ind[:, j]), axis = 1)
            u_index = tf.stack((tf.range(self.batch_size), u_ind[:, j]), axis = 1)
            
            main_Q_distrib_l = tf.gather_nd(self.Q_distrib_given_actions, l_index)
            main_Q_distrib_u = tf.gather_nd(self.Q_distrib_given_actions, u_index)
            
            main_Q_distrib_l = tf.clip_by_value(main_Q_distrib_l, 1e-10, 1.0)
            main_Q_distrib_u = tf.clip_by_value(main_Q_distrib_u, 1e-10, 1.0)
            
            critic_loss += self.Q_distrib_next[:, j] * (
                    (u[:, j] - bj[:,j]) * tf.log(main_Q_distrib_l) +
                    (bj[:, j] - l[:,j]) * tf.loc(main_Q_distrib_u))
            
        # Calculate the mean loss on the batch
        critic_loss  = tf.negative(critic_loss)
        critic_loss  = tf.reduct_mean(critic_loss)
        critic_loss += l2_regularization(self.critic_parameters) # penalize the critic for having large weights -> L2 Regularization
        
        ##############################################################################
        ##### Develop the Operation that Trains the critic with Gradient Descent #####
        ##############################################################################
        critic_trainer             = tf.train.AdamOptimizer(Settings.CRITIC_LEARNING_RATE)
        self.train_critic_one_step = critic_trainer.minimize(critic_loss) # RUN THIS TO TRAIN THE CRITIC
        
        
        ##############################################################################
        ##### Develop the Operation that trains the actor with Gradient Descent ###### Note: dQ/dActor_parameter = (dQ/dAction)*(dAction/dActor_parameter)
        ##############################################################################
        self.dQ_dAction           = tf.gradients(self.q_network_value_with_actor_for_action, self.actor)[0] # also called 'action_gradients'
        self.actor_gradiants      = tf.gradients(self.actor, self.actor_parameters, -self.dQ_dAction)       # pushing the gradients through to the actor parameters
        actor_trainer             = tf.train.AdamOptimizer(Settings.ACTOR_LEARNING_RATE)                    # establishing the training method
        self.train_actor_one_step = actor_trainer.apply_gradients(zip(self.actor_grad, self.actor_vars))    # RUN THIS TO TRAIN THE ACTOR
        
        
        
        
        
    def run(self):
        # Continuously train the actor and the critic, by applying gradient
        # descent to batches of data sampled from the replay buffer
        
        self.total_training_iterations = 1 # initializing the counter of training iterations
        start_time = time.time()
        
        with self.sess.as_default(), self.sess.graph.as_default(): # pulling the tensorflow session here
            
            self.sess.run(self.initialize_target_network_parameters)
            
            # Continuously train...
            while True:
                
                # Sample mini-batch of data from the replay buffer
                sampled_batch = np.asarray(self.replay_buffer.sample())
                
                # Assemble this data into a dictionary that will be used for training
                training_data_dict = {self.state_placeholder: np.stack(sampled_batch[:, 0]),
                             self.action_placeholder: np.stack(sampled_batch[:, 1]),
                             self.reward_placeholder: sampled_batch[:, 2],
                             self.next_state_placeholder: np.stack(sampled_batch[:, 3]),
                             self.not_done_placeholder: sampled_batch[:, 4]}
                
                ##################################
                ##### TRAIN ACTOR AND CRITIC #####
                ##################################
                self.sess.run([self.train_critic_one_step, self.train_actor_one_step], feed_dict = training_data_dict)
                
                
                # If it's time to update the target networks
                if self.total_training_iterations % Settings.UPDATE_TARGET_NETWORKS_EVERY_NUM_ITERATIONS == 0:
                    self.sess.run(self.update_target_network_parameters)
                    
                self.total_training_iterations += 1
                    
                    
                # If it's time to print the training performance to the screen
                if self.total_training_iterations % Settings.DISPLAY_PERFORMANCE_EVERY_NUM_ITERATIONS == 0:
                    print("Trained actor and critic %i iterations in %f s" % (Settings.PERFORMANCE_UPDATE_EVERY_NUM_ITERATIONS, time.time() - start_time))
                    start_time = time.time() # resetting the timer for the next PERFORMANCE_UPDATE_EVERY_NUM_ITERATIONS of iterations