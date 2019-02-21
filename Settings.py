""" 
    All settings relating to D4PG are contained in this file.
    This file is copied on each run and placed in the Tensorboard directory
    so that all settings are preserved for future reference. 
"""

class Settings:
    
    #%% 
    ########################
    ##### Run Settings #####
    ########################
    
    RUN_NAME               = 'LunarLander' # use just the name. If trying to restore from file, use name along with timestamp
    USE_GYM                = 1 # 0 = use (your own) dynamics; 1 = use openAI's gym (for testing)
    ENVIRONMENT            = 'LunarLanderContinuous-v2'
    #ENVIRONMENT            = 'Pendulum-v0'
    RECORD_VIDEO           = True
    VIDEO_RECORD_FREQUENCY = 1000 # Multiples of "CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES"
    LEARN_FROM_PIXELS      = False # False = learn from state (fully observed); True = learn from pixels (partially observed)
    RESUME_TRAINING        = False # If True, be sure to set "RUN_NAME" to the previous run's filename
    USE_GPU_WHEN_AVAILABLE = True # As of Nov 19, 2018, it appears better to use CPU. Re-evaluate again later
    RANDOM_SEED            = 13
    
    #%% 
    #############################
    ##### Training Settings #####
    #############################
    
    # Hyperparameters
    NUMBER_OF_ACTORS        = 4
    NUMBER_OF_EPISODES      = 1e5 # that each agent will perform
    MAX_TRAINING_ITERATIONS = 3e5
    MAX_NUMBER_OF_TIMESTEPS = 1000 # per episode
    ACTOR_LEARNING_RATE     = 0.0001
    CRITIC_LEARNING_RATE    = 0.0001
    TARGET_NETWORK_TAU      = 0.001
    DISCOUNT_FACTOR         = 0.99
    N_STEP_RETURN           = 5
    NUMBER_OF_BINS          = 51 # Also known as the number of atoms
    NORMALIZE_STATE         = False
    REWARD_SCALING          = 100.0 # Amount to scale down the reward signal
    MIN_Q                   = -4.0
    MAX_Q                   = 2.0
    L2_REGULARIZATION       = False # optional for training the critic
    L2_REG_PARAMETER        = 1e-6
    
    # Periodic events
    UPDATE_TARGET_NETWORKS_EVERY_NUM_ITERATIONS       = 1 
    UPDATE_ACTORS_EVERY_NUM_EPISODES                  = 1
    CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES       = 5    
    LOG_TRAINING_PERFORMANCE_EVERY_NUM_ITERATIONS     = 100
    DISPLAY_TRAINING_PERFORMANCE_EVERY_NUM_ITERATIONS = 25000
    DISPLAY_ACTOR_PERFORMANCE_EVERY_NUM_EPISODES      = 5000
    
    # Buffer settings
    PRIORITY_REPLAY_BUFFER = False
    PRIORITY_ALPHA         = 0.6            # Controls the randomness vs prioritisation of the prioritised sampling (0.0 = Uniform sampling, 1.0 = Greedy prioritisation)
    PRIORITY_BETA_START    = 0.4       # Starting value of beta - controls to what degree IS weights influence the gradient updates to correct for the bias introduced by priority sampling (0 - no correction, 1 - full correction)
    PRIORITY_BETA_END      = 1.0         # Beta will be linearly annealed from its start value to this value throughout training
    PRIORITY_EPSILON       = 0.00001      # Small value to be added to updated priorities to ensure no sample has a probability of 0 of being chosen
    DUMP_PRIORITY_REPLAY_BUFFER_EVER_NUM_ITERATIONS = 200 # Check if the prioritized replay buffer is overfulled every ## iterations. If so, dump the excess data
    
    REPLAY_BUFFER_SIZE                    = 1000000
    REPLAY_BUFFER_START_TRAINING_FULLNESS = 0 # how full the buffer should be before training begins
    MINI_BATCH_SIZE                       = 256
        
    # Exploration noise
    UNIFORM_OR_GAUSSIAN_NOISE = False # True -> Uniform; False -> Gaussian
    if UNIFORM_OR_GAUSSIAN_NOISE:
        NOISE_SCALE           = 1 # 1 is best for uniform -> noise scaled to the action range
    else:
        NOISE_SCALE           = 0.3 # 0.3 is better for gaussian -> since normal distribution can be large
        #NOISE_SCALE = 0.
    NOISE_SCALE_DECAY         = 0.9999 # 1 means the noise does not decay during training
    
#%%
    ####################################
    ##### Model Structure Settings #####
    ####################################
    
    # Whether or not to learn from pixels (defined above)
    if LEARN_FROM_PIXELS:
        # Define the properties of the convolutional layer in a list. Each dict in the list is one layer
        # 'filters' gives the number of filters to be used
        # 'kernel_size' gives the dimensions of the filter
        # 'strides' gives the number of pixels that the filter skips while colvolving
        CONVOLUTIONAL_LAYERS =  [{'filters': 32, 'kernel_size': [8, 8], 'strides': [4, 4]},
                                 {'filters': 64, 'kernel_size': [4, 4], 'strides': [2, 2]},
                                 {'filters': 64, 'kernel_size': [3, 3], 'strides': [1, 1]}]
        
        
    # Fully connected layers follow the (optional) convolutional layers        
    ACTOR_HIDDEN_LAYERS  = [400, 300] # number of hidden neurons in each layer
    CRITIC_HIDDEN_LAYERS = [400, 300] # number of hidden neurons in each layer

    #%%     
    ##############################
    #### Environment Settings ####
    ##############################    
    # Get state & action shapes from environment & action bounds
    if USE_GYM:
        import gym
        test_env_to_get_settings = gym.make(ENVIRONMENT)
        
        if LEARN_FROM_PIXELS:
            STATE_SIZE = [84, 84, 4] # dimension of the pixels that are used as the observation/state
        else:
            STATE_SIZE = list(test_env_to_get_settings.observation_space.shape) # dimension of the observation/state space
            
        ACTION_SIZE          = test_env_to_get_settings.action_space.shape[0] # dimension of the action space
        LOWER_ACTION_BOUND   = test_env_to_get_settings.action_space.low # lowest action for each action [action1, action2, action3, etc.]
        UPPER_ACTION_BOUND   = test_env_to_get_settings.action_space.high # highest action for each action [action1, action2, action3, etc.]
        UPPER_STATE_BOUND    = test_env_to_get_settings.observation_space.high # highest state we will encounter along each dimension
    
        del test_env_to_get_settings # getting rid of this test environment

    #########################
    #### TO BE COMPLETED ####
    #########################        
    else: # use your own dynamics
        from Dynamics import Dynamics
        
        if LEARN_FROM_PIXELS:
            STATE_SIZE     = 0 # INCOMPLETE
        else:
            STATE_SIZE     = 0 # INCOMPLETE
        ACTION_SIZE        = 0 # INCOMPLETE
        LOWER_ACTION_BOUND = 0 # INCOMPLETE
        UPPER_ACTION_BOUND = 0 # INCOMPLETE
        
        
    ACTION_RANGE = UPPER_ACTION_BOUND - LOWER_ACTION_BOUND # range for each action
        
    #%% 
    #########################
    ##### Save Settings #####
    #########################
    
    MODEL_SAVE_DIRECTORY                 = 'Tensorboard/' # where to save all data
    TENSORBOARD_FILE_EXTENSION           = '.tensorboard' # file extension for tensorboard file
    SAVE_CHECKPOINT_EVERY_NUM_ITERATIONS = 50000 # how often to save the neural network parameters
