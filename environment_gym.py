
"""
This script generates the environment for reinforcement learning agents.

OpenAI's gym environments are used here to test the performance of the C51
learning algorithm.

All dynamic environments I create will have a standardized architecture. The 
reason for this is I have one learning algorithm and many environments. All 
environments are responsible for:
    - dynamics propagation (via the step method)
    - initial conditions   (via the reset method)
    - reporting environment properties (defined in __init__)
    - seeding the dynamics (via the seed method)
          
Outputs:
    Reward must be of shape ()
    State must be of shape (state_size,)
    Done must be a bool

Inputs:
    Action input is of shape (action_size,)   


@author: Kirk Hovell (khovell@gmail.com)
"""
import numpy as np
import signal
import multiprocessing
import gym

class Environment:
    
    def __init__(self): 
        ##################################
        ##### Environment Properties #####
        ##################################
        self.ENVIRONMENT = 'LunarLander-v2'        
        self.env = gym.make(self.ENVIRONMENT)
        
        self.STATE_SIZE           = list(self.env.observation_space.shape)[0] # dimension of the observation/state space            
        self.ACTION_SIZE          = self.env.action_space.n # dimension of the action space
        self.UPPER_STATE_BOUND    = self.env.observation_space.high # highest state we will encounter along each dimension
        self.TIMESTEP = 0.1
        
        
    ###################################
    ##### Seeding the environment #####
    ###################################
    def seed(self, seed):
        np.random.seed(seed)   
        self.env.seed(seed)
        
        
    ######################################
    ##### Resettings the Environment #####
    ######################################    
    def reset(self):
        # This method resets the state and returns it
        
        self.state = self.env.reset()
        
        # Return the state
        return self.state
        
    #####################################
    ##### Step the Dynamics forward #####
    #####################################
    def step(self, action):
        
        self.state, reward, done, _ = self.env.step(action)
        
        # Return the (state, reward, done)
        return self.state, reward, done


    #########################################################################
    ##### Generating communication links between environment and agents #####
    #########################################################################
    def generate_queue(self):
        # Generate the queues responsible for communicating with the agent
        self.agent_to_env = multiprocessing.Queue(maxsize = 1)  
        self.env_to_agent = multiprocessing.Queue(maxsize = 1)  
        return self.agent_to_env, self.env_to_agent
        
    

    ###################################
    ##### Running the environment #####
    ###################################    
    def run(self):

        """
        This method is called when the environment process is launched by main.py.
        It is responsible for continually listening for an input action from the
        agent through a Queue. If an action is received, it is to step the environment
        and return the results.
        """
        
        # Instructing this process to treat Ctrl+C events (called SIGINT) by going SIG_IGN (ignore).
        # This permits the process to continue upon a Ctrl+C event to allow for graceful quitting.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        # Loop until the process is terminated
        while True:
            # Blocks until the agent passes us an action
            action = self.agent_to_env.get()
            
            if type(action) == bool:                
                # The signal to reset the environment was received
                state = self.reset()
                # Return the results
                self.env_to_agent.put(state)
            
            else:            
                ################################
                ##### Step the environment #####
                ################################
                next_state, reward, done = self.step(action)
                
                # Return the results
                self.env_to_agent.put((next_state, reward, done))
        
