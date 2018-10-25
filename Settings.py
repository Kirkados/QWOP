
from multiprocessing import cpu_count

class Settings:
    
    #%% Environment Settings 
    ENVIRONMENT = 'cartpole'
    LEARN_FROM_PIXELS = 0 # 0 = learn from state (fully observed); 1 = learn from pixels (partially observed)
    
    
    #%% Training Settings
    NUMBER_OF_EPISODES      = 10
    MAX_NUMBER_OF_TIMESTEPS = 50
    
    UPDATE_TARGET_NETWORKS_EVERY_NUM_ITERATIONS = 1
    DISPLAY_PERFORMANCE_EVERY_NUM_ITERATIONS    = 10
    
    REPLAY_BUFFER_SIZE = 1e7
    MINI_BATCH_SIZE = 32
    PRIORITIZED_REPLAY_BUFFER = True
    

    
    
    
    #%% Network Settings
    
    NUMBER_OF_ACTORS = cpu_count() - 2 # ideally, cpu_count() - 2 (no GPU) OR cpu_count() - 1 (with GPU)
    
    # Whether or not to learn from pixels (defined above)
    if LEARN_FROM_PIXELS:
        # Define the properties of the convolutional layer in a list. Each dict in the list is one layer
        # 'filters' gives the number of filters to be used
        # 'kernel_size' gives the dimensions of the filter
        # 'strides' gives the number of pixels that the filter skips while colvolving
        CONVOLUTIONAL_LAYERS =  [{'filters': 32, 'kernel_size': [8, 8], 'strides': [4, 4]},
                                 {'filters': 64, 'kernel_size': [4, 4], 'strides': [2, 2]},
                                 {'filters': 64, 'kernel_size': [3, 3], 'strides': [1, 1]}]
        
        
    ACTOR_HIDDEN_LAYERS  = [100, 100, 30] # number of hidden neurons in each layer
    CRITIC_HIDDEN_LAYERS = [100, 100, 30] # number of hidden neurons in each layer
    
    MIN_Q = -1000
    MAX_Q = 1000
    
    
    
    #%% Hyperparameters
    DISCOUNT_FACTOR      = 0.99
    NUMBER_OF_ATOMS      = 51 # WHAT ARE YOU?!?! NUMBER OF ATOMS FOR THE CRITIC?!?!?!?
    ACTOR_LEARNING_RATE  = 5e-4
    CRITIC_LEARNING_RATE = 5e-4
    TARGET_NETWORK_TAU   = 0.001
    N_STEP_RETURN = 5
    
    
    #%% Environment Settings
    
    ###########################
    #### Dynamics selector ####
    ###########################
    USE_GYM = 0 # 0 = use (your own) dynamics; 1 = use openAI's gym (for testing)
    
    
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
    
        del test_env_to_get_settings # getting rid of this test environment
        
        
        
        
        #########################
        #### TO BE COMPLETED ####
        #########################
        
    else: # use your own dynamics
        from Dynamics import Dynamics
        
        if LEARN_FROM_PIXELS:
            STATE_SIZE     = 0
        else:
            STATE_SIZE     = 0
        ACTION_SIZE        = 0
        LOWER_ACTION_BOUND = 0
        UPPER_ACTION_BOUND = 0
        
    ACTION_RANGE         = UPPER_ACTION_BOUND - LOWER_ACTION_BOUND # range for each action
        
    #%% Save Settings
    
    MODEL_SAVE_DIRECTORY = 'saved_models/'
    TRY_TO_LOAD_IN_PARAMETERS = True
    
    
    #%% Asking the environment for information regarding action size & bounds
