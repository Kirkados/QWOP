"""
This code produces the actor/agent that will execute episodes.

The agent calculates the N-step returns and dumps data into the ReplayBuffer.
It collects its updated parameters after each episode. 
"""

import tensorflow as tf
import numpy as np

from settings import Settings
from collections import deque
from build_neural_networks import build_actor_network
from neural_network_utilities import get_variables, copy_variables

class Agent:
    
    def __init__(self, sess, n_agent, displayer, replay_buffer):
        
        print("Initializing agent " + str(n_agent))
        
        self.n_agent = n_agent
        self.sess = sess
        self.replay_buffer = replay_buffer
        
        self.create_summary_functions()
    
    def create_summary_functions(self):
        self.episode_reward_placeholder  = tf.placeholder(tf.float32)
        self.noise_placeholder           = tf.placeholder(tf.float32)
        self.timestep_number_placeholder = tf.placeholder(tf.float32)
        
        episode_reward_summary              = tf.summary.scalar("Episode/Episode reward", self.episode_reward_placeholder)
        noise_placeholder_summary           = tf.summary.scalar("Episode/Noise", self.noise_placeholder)
        timestep_number_placeholder_summary = tf.summary.scalar("Episode/Number_of_timesteps", self.timestep_number_placeholder)
        
        self.episode_summary = tf.summary.merge([episode_reward_summary,
                                                 noise_placeholder_summary,
                                                 timestep_number_placeholder_summary])
    
        self.writer = tf.summary.FileWriter(f"./logs/Agent_{self.n_agent}", self.sess.graph)
    
    def build_actor(self):
        # Build the actor's policy neural network
        agent_name = 'agent_' + str(self.n_agent) # worker name 'agent_3' for example
        self.state_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, *Settings.STATE_SIZE], name = 'state_placeholder')
        
        
        #############################
        #### Generate this Actor ####
        #############################
        self.policy = build_actor_network(self.state_placeholder, trainable = False, scope = agent_name) # using the actor generating function from BuildNeuralNetworks.py
        
        # Getting the non-trainable parameters from this actor, so that we will know where to place the updated parameters
        self.variables = get_variables(scope = agent_name, trainable = False) 
    
    
    def choose_action(self, state):
        # Accepts the state, runs it through the neural network, and returns the properly scaled action.
        
        ########################
        #### Run the Policy ####
        ########################
        return self.sess.run(self.polcy, feed_dict = {self.state_placeholder: state[None]})[0]
    
    def update_actor_parameters(self):
        # Grab the most up-to-date parameters for the actor neural network. The parameters come from the learner (who is constantly learning on the GPU)
        
        with self.sess.as_default(), self.sess.graph.as_default(): # make sure the tf.Session() is available here.
            
            # get the up-to-date parameters from the actor that is being trained
            self.newly_trained_parameters = get_variables('learned_actor', trainable = True) 
            
            # copy the learned variables to the actor
            copy_variables(self.newly_trained_parameters, self.variables, 1)
            
        #### NOTE: IF PERFORMANCE IS BAD, POSSIBLY instead write: self.update = copy_variables(self.newly_trained_parameters, self.variables, 1) and then just call self.update when needed.

    
    def run(self):
        # Runs the agent in its own environment
        # Runs for a specified number of episodes, or until told to stop
        
        print("Starting to run agent " + str(self.n_agent))
        
        # Initializing parameters for agent network
        self.update_actor_parameters()
        
        self.episode_number = 1
        self.timestep_number = 0
        
        # For all requested episodes
        for episode_number in range(Settings.NUMBER_OF_EPISODES):
            
            ####################################
            #### Getting this episode ready ####
            ####################################
            
            # Resetting the environment for this episode
            state = self.env.reset()
            
            # Creating the n-step temporary memory
            n_step_memory = deque()
            
            # Resetting items for this episode
            episode_reward = 0
            timestep_number = 0  
            done = False
            
            # For each timestep in each episode
            while timestep_number < Settings.MAX_NUMBER_OF_TIMESTEPS and not done:
                
                # Getting action by running neural network and clipping it to be within limits
                action = np.clip(self.choose_action(state), Settings.LOWER_ACTION_BOUND, Settings.UPPER_ACTION_BOUND)
                
                # Calculating gaussian exploration noise
                noise = np.multiply(np.random.normal(size = Settings.ACTION_SIZE),Settings.ACTION_RANGE)
                
                # Add exploration noise to original action, and clipping it again
                action = np.clip(action + noise, Settings.LOWER_ACTION_BOUND, Settings.UPPER_ACTION_BOUND)
                
                ################################################
                #### Step the dynamics forward one timestep ####
                ################################################
                next_state, reward, done, _ = self.env.step(action)
                
                # Add reward we just received to running total
                episode_reward += reward
                                
                # Store the data in this temporary buffer until we calculate the n-step return
                n_step_memory.append((state, action, reward))
                
                if len(n_step_memory) >= Settings.N_STEP_RETURN:
                    state_memory, action_memory, n_step_reward = n_step_memory.popleft()
                    for i, (state_i, action_i, reward_i) in enumerate(n_step_memory):
                        n_step_reward += reward_i*Settings.DISCOUNT_FACTOR**(i + 1)
                    self.replay_buffer.add(state_memory, action_memory, n_step_reward, next_state, done) # dump data into large replay buffer
                    
                # End of timestep -> next state becomes current state
                state = next_state
                timestep_number += 1
                
                ##########################
                #### EPISODE COMPLETE ####
                ##########################
                
            # Periodically update the agent with the GPU's most recent version of the network parameters
            if self.episode_number % Settings.UPDATE_ACTORS_EVERY_NUM_EPISODES == 0:
                self.update_actor_parameters()
            
            # Plot stuff here if desired... I'll use tensorboard instead #
            
            # Write data to tensorboard summary
            feed_dict = {self.episode_reward_placeholder:  episode_reward,
                         self.noise_placeholder:           noise,
                         self.timestep_number_placeholder: timestep_number}
            summary = self.sess.run(self.episode_summary, feed_dict=feed_dict)
            self.writer.add_summary(summary, self.episode_number)
            
        self.env.close()