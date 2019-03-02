"""
This Agent class generates one agent that will run episodes. The agent collects, 
processes, and dumps data into the ReplayBuffer. It will occasionally update the 
parameters used by its neural network by grabbing the most up-to-date ones from 
the Learner. 

The environment is not contained in this thread because it must be in its own
process. The agent communicates with the environment through two queues:
    agent_to_env - the agent passes actions or reset signals to the environment process
    env_to_agent - the environment returns results to the agent

@author: Kirk Hovell (khovell@gmail.com)
"""

import time
import tensorflow as tf
import numpy as np
from collections import deque
from pyvirtualdisplay import Display # for rendering

from settings import Settings
from build_neural_networks import BuildQNetwork
environment_file = __import__('environment_' + Settings.ENVIRONMENT) # importing the environment


class Agent:
    
    def __init__(self, sess, n_agent, agent_to_env, env_to_agent, replay_buffer, writer, filename, learner_policy_parameters):
        
        print("Initializing agent " + str(n_agent) + "...")
        
        # Saving inputs to self object for later use
        self.n_agent = n_agent
        self.sess = sess
        self.replay_buffer = replay_buffer
        self.learner_policy_parameters = learner_policy_parameters
        self.agent_to_env = agent_to_env
        self.env_to_agent = env_to_agent
        
        # Record video if desired
        if self.n_agent == 1 and Settings.RECORD_VIDEO:
            # Generates a fake display that collects the frames during rendering.
            # Must be rendered on Linux.
            try:
                display = Display(visible = 0, size = (1400,900))
                display.start()
            except:
                print("You must run on Linux if you want to record gym videos!")
                    
        # Build this Agent's actor network
        self.build_q_network()
        
        # Build the operations to update the actor network
        self.build_q_network_update_operation()
        
        # Establish the summary functions for TensorBoard logging.
        self.create_summary_functions()        
        self.writer = writer
        
        print("Agent %i initialized!" % self.n_agent)
    
    
    def create_summary_functions(self):            
        # Logging the timesteps used for each episode for each agent
        self.timestep_number_placeholder      = tf.placeholder(tf.float32)            
        timestep_number_summary               = tf.summary.scalar("Agent_" + str(self.n_agent) + "/Number_of_timesteps", self.timestep_number_placeholder)
        self.regular_episode_summary          = tf.summary.merge([timestep_number_summary])
        
        # If this is agent 1, the agent who will also test performance, additionally log the reward
        if self.n_agent == 1:
            self.episode_reward_placeholder   = tf.placeholder(tf.float32)
            test_time_episode_reward_summary  = tf.summary.scalar("Test_agent/Episode_reward", self.episode_reward_placeholder)
            test_time_timestep_number_summary = tf.summary.scalar("Test_agent/Number_of_timesteps", self.timestep_number_placeholder)
            self.test_time_episode_summary    = tf.summary.merge([test_time_episode_reward_summary, test_time_timestep_number_summary])
            
            
    def build_q_network(self):        
        # Generate the actor's policy neural network
        agent_name = 'agent_' + str(self.n_agent) # agent name 'agent_3', for example
        self.state_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, Settings.STATE_SIZE], name = 'state_placeholder') # the * lets Settings.STATE_SIZE be not restricted to only a scalar
                
        #############################
        #### Generate this Actor ####
        #############################
        self.policy = BuildQNetwork(self.state_placeholder, scope = agent_name)


    def build_actor_update_operation(self):
        # Update agent's policy network parameters from the most up-to-date version from the learner
        update_operations = []
        source_variables = self.learner_policy_parameters
        destination_variables = self.policy.parameters
        
        # For each parameters in the network
        for source_variable, destination_variable in zip(source_variables, destination_variables):
            # Directly copy from the learner to the agent
            update_operations.append(destination_variable.assign(source_variable))
        
        # Save the operation that performs the actor update
        self.update_actor_parameters = update_operations
            
    def run(self, stop_run_flag, replay_buffer_dump_flag, starting_episode_number):        
        # Runs the agent in its own environment
        # Runs for a specified number of episodes or until told to stop
        print("Starting to run agent %i at episode %i." % (self.n_agent, starting_episode_number[self.n_agent -1]))
        
        # Initializing parameters for agent network
        self.sess.run(self.update_actor_parameters)
        
        # Getting the starting episode number. If we are restarting a training
        # run that has crashed, the starting episode number will not be 1.
        episode_number = starting_episode_number[self.n_agent - 1] 
        
        # Start time
        start_time = time.time()
        
        # Creating the temporary memory space for calculating N-step returns
        self.n_step_memory = deque()
        
        # For all requested episodes or until user flags for a stop (via Ctrl + C)        
        while episode_number <= Settings.NUMBER_OF_EPISODES and not stop_run_flag.is_set(): 
            ####################################
            #### Getting this episode ready ####
            ####################################            
            # Resetting the environment for this episode by sending a boolean
            self.agent_to_env.put(False)
            state = self.env_to_agent.get()

            # Normalizing the state to 1 separately along each dimension
            # to avoid the 'vanishing gradients' problem
            if Settings.NORMALIZE_STATE:
                state = state/Settings.UPPER_STATE_BOUND
            
            # Clearing the N-step memory for this episode
            self.n_step_memory.clear()
            
            # Checking if this is a test time (when we run an agent in a 
            # noise-free environment to see how the training is going).
            # Only agent_1 is used for test time
            test_time = (self.n_agent == 1) and (episode_number % Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES == 0)
            
            # Calculating the noise scale for this episode. The noise scale 
            # allows for changing the amount of noise added to the actor during training.
            if test_time:
                # It's test time! Run this episode without noise to evaluate performance.
                exploration_rate = 0.           
                
                # Additionally, if it's time to render, make a statement to the user
                if Settings.RECORD_VIDEO and episode_number % (Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES*Settings.VIDEO_RECORD_FREQUENCY) == 0:
                    print("Rendering Actor %i at episode %i" % (self.n_agent, episode_number))
                    
                    # Also log the states & actions encountered in this episode because we are going to render them!
                    state_log = []
                    action_log = []
                    time_log = []
                
            else:
                # Regular training episode, use noise.
                # Noise is decayed during the training
                exploration_rate = 0.1
            
            # Resetting items for this episode
            episode_reward = 0
            timestep_number = 0
            done = False
            
            # Stepping through time until we reach the max time length or until episode completes.
            while timestep_number < Settings.MAX_NUMBER_OF_TIMESTEPS and not done:
                
                # Should we try explore on this timestep?
                if np.random.uniform() < exploration_rate:
                    # Time to explore! 
                    action = np.random.randint(low = 0, high = Settings.ACTION_SIZE)
                else:
                    # Do what you think is best
                    ##############################
                    ##### Running the Policy #####
                    ##############################
                    q_distribution = self.sess.run(self.policy.q_distribution, feed_dict = {self.state_placeholder: np.expand_dims(state,0)})[0] # [# actions, # bins]
                    # Collapsing the distributions into q_values corresponding to each action
                    q_values = np.sum(self.policy.bins * q_distribution, axis = 1) # [# actions, 1]                    
                    # Choosing the most lucrative action
                    action = np.argmax(q_values, axis = 0) # [1]
                

                ################################################
                #### Step the dynamics forward one timestep ####
                ################################################
                # Send the action to the environment process
                self.agent_to_env.put(action)                
                # Receive results from stepped environment
                next_state, reward, done = self.env_to_agent.get()
                
                # Add reward we just received to running total for this episode
                episode_reward += reward
                
                # Normalize the state and scale down reward
                if Settings.NORMALIZE_STATE:
                    next_state = next_state/Settings.UPPER_STATE_BOUND
                reward = reward/Settings.REWARD_SCALING
                                
                # Store the data in this temporary buffer until we calculate the n-step return
                self.n_step_memory.append((state, action, reward))
                
                # If the n-step memory is full enough and we aren't testing performance
                if (len(self.n_step_memory) >= Settings.N_STEP_RETURN) and not test_time: 
                    # Grab the oldest data from the n-step memory
                    state_0, action_0, reward_0 = self.n_step_memory.popleft()                     
                    # N-step reward starts with reward_0
                    n_step_reward = reward_0                     
                    # Initialize gamma
                    discount_factor = Settings.DISCOUNT_FACTOR                     
                    for (state_i, action_i, reward_i) in self.n_step_memory:
                        # Calculate the n-step reward
                        n_step_reward += reward_i*discount_factor
                        discount_factor *= Settings.DISCOUNT_FACTOR # for the next step, gamma**(i+1)
                        
                    # Dump data into large replay buffer
                    # If the prioritized replay buffer is currently dumping data, 
                    # wait until that is done before adding more data to the buffer
                    replay_buffer_dump_flag.wait() # blocks until replay_buffer_dump_flag is True
                    self.replay_buffer.add((state_0, action_0, n_step_reward, next_state, done, discount_factor))
                
                # If this episode is being rendered, log the state for rendering later
                if self.n_agent == 1 and Settings.RECORD_VIDEO and episode_number % (Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES*Settings.VIDEO_RECORD_FREQUENCY) == 0:                    
                    state_log.append(state)
                    action_log.append(action)
                    time_log.append(timestep_number*Settings.TIMESTEP)
                
                # End of timestep -> next state becomes current state
                state = next_state
                timestep_number += 1
                                
                # If this episode is done, drain the N-step buffer, calculate 
                # returns, and dump in replay buffer unless it is test time.
                if (done or (timestep_number == Settings.MAX_NUMBER_OF_TIMESTEPS)) and not test_time:                    
                    # Episode has just finished, calculate the remaining N-step entries
                    while len(self.n_step_memory) > 0:
                        # Grab the oldest data from the n-step memory
                        state_0, action_0, reward_0 = self.n_step_memory.popleft()                         
                        # N-step reward starts with reward_0
                        n_step_reward = reward_0 
                        # Initialize gamma
                        discount_factor = Settings.DISCOUNT_FACTOR 
                        for (state_i, action_i, reward_i) in self.n_step_memory:                            
                            # Calculate the n-step reward
                            n_step_reward += reward_i*discount_factor
                            discount_factor *= Settings.DISCOUNT_FACTOR # for the next step, gamma**(i+1)
                        
                        
                        # dump data into large replay buffer
                        replay_buffer_dump_flag.wait()
                        self.replay_buffer.add((state_0, action_0, n_step_reward, next_state, done, discount_factor))
                        
            
            ################################
            ####### Episode Complete #######
            ################################       
            # If this episode is being rendered, render it now.
            if self.n_agent == 1 and Settings.RECORD_VIDEO and episode_number % (Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES*Settings.VIDEO_RECORD_FREQUENCY) == 0:                    
                environment_file.render(np.asarray(state_log), np.asarray(action_log), time_log, episode_number, Settings.RUN_NAME)
                
            # Periodically update the agent with the learner's most recent version of the actor network parameters
            if episode_number % Settings.UPDATE_ACTORS_EVERY_NUM_EPISODES == 0:
                self.sess.run(self.update_actor_parameters)
            
            # Periodically print to screen how long it's taking to run these episodes
            if episode_number % Settings.DISPLAY_ACTOR_PERFORMANCE_EVERY_NUM_EPISODES == 0:
                print("Actor " + str(self.n_agent) + " ran " + str(Settings.DISPLAY_ACTOR_PERFORMANCE_EVERY_NUM_EPISODES) + " episodes in %.1f minutes, and is now at episode %i" % ((time.time() - start_time)/60, episode_number))
                start_time = time.time()

            ###################################################
            ######## Log training data to tensorboard #########
            ###################################################
            # Write training data to tensorboard if we just checked the greedy performance of the agent (i.e., if it's test time).
            if test_time:                  
                # Write test to tensorboard summary -> reward and number of timesteps
                feed_dict = {self.episode_reward_placeholder:  episode_reward,
                             self.timestep_number_placeholder: timestep_number}                
                summary = self.sess.run(self.test_time_episode_summary, feed_dict = feed_dict)
                self.writer.add_summary(summary, episode_number)
            else: 
                # it's not test time -> simply write the number of timesteps taken (reward is irrelevant)
                feed_dict = {self.timestep_number_placeholder: timestep_number}                
                summary = self.sess.run(self.regular_episode_summary, feed_dict = feed_dict)
                self.writer.add_summary(summary, episode_number)                
            
            # Increment the episode counter
            episode_number += 1
        
        #################################
        ##### All episodes complete #####
        #################################
        # Notify of completion
        print("Actor %i finished after running %i episodes!" % (self.n_agent, episode_number - 1))