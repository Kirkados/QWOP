"""
This code is used to generate and use the actors and critics used in DDPG
"""

import tensorflow as tf
import KH_nn_utils
import numpy as np

# Class contains all the workings of the policy network, including building, training, and executing
class PolicyNetwork():
    """
    Policy network receives the current state and outputs an action deterministically.
    Output is bounded by a tanh to limit the allowable torque on the agent
    """
    # __init__ runs automatically the first time PolicyNetwork() is called
    def __init__(self, sess, n_hidden_layers, n_neurons, torque_limit, state_dim, action_dim, learning_rate, batch_size, target_update_rate, hidden_activation):
        self.sess = sess
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.torque_limit = torque_limit
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size # size of mini-batches used in training
        self.tau = target_update_rate
        self.hidden_activation = hidden_activation
        
        # Create the policy (actor) network
        self.current_state, self.action_out = self.create_actor_network()
        
        # Grabbing all trainable parameters of network
        self.actor_network_parameters = tf.trainable_variables()
        
        # Create the policy target network
        self.target_current_state, self.target_action_out = self.create_actor_network()
        
        # Getting the target policy parameters
        self.target_actor_network_parameters = tf.trainable_variables()[len(self.actor_network_parameters):]

        # Operation to update the target policy with the real policy weights. Loops through all parameters -> new_target = tau*real + (1-tau)*target
        self.update_target_actor_network_parameters = [self.target_actor_network_parameters[i].assign(tf.multiply(self.actor_network_parameters[i],
                                                       self.tau) + tf.multiply(self.target_actor_network_parameters[i], 1. - self.tau))
                                                       for i in range(len(self.target_actor_network_parameters))]
        
        
        
        # Placeholder for action gradients of Q function from critic
        self.action_Q_gradients = tf.placeholder(tf.float32, [None, self.action_dim])
        
        # Calculating gradients (Delta_a Q * Delta_theta Pi)
        self.unnormalized_actor_gradients = tf.gradients(self.action_out, self.actor_network_parameters, -self.action_Q_gradients)
        
        # Dividing by the batch size (normalizing kind of)
        self.actor_gradients = [tf.div(x, self.batch_size) for x in self.unnormalized_actor_gradients]
        #self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        
        # Selecting Optimizer, specifying how to apply the gradients we calculated (gradient, parameter to update)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_actor_one_step = self.optimizer.apply_gradients(zip(self.actor_gradients, self.actor_network_parameters))
        
        # Total number of trainable variables from the actor & actor target
        self.num_trainable_variables = len(self.actor_network_parameters) + len(self.target_actor_network_parameters)
        
        #####################################################################################
        ###****** COULD ALL THIS BE APPLIED SIMPLER WITH THE USE OF A LOSS FUNCTION ******###
        #####################################################################################
        
    # Creates the policy network and returns the state placeholder and output actions
    def create_actor_network(self):
        current_state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name = "current_state") # placeholder for input to policy NN
                
        # Building n layer policy network, with a bounded action space        
        layer_activations = {} # initializing the dictionary that will hold the outputs of each layer
        layer_activations["layer0"] = current_state
        for i in range(self.n_hidden_layers): # for the first to the second last layer
            _, layer_activations["layer" + str(i+1)] = KH_nn_utils.fc_layer(layer_activations["layer" + str(i)], input_size =int(layer_activations["layer" + str(i)].get_shape()[1]), output_size = self.n_neurons, name = "FC" + str(i+1), nonlinear = self.hidden_activation, batch_norm = True)
            
        # for the final layer, which has an output of the size of the actions and is constrained
        _, raw_action_output = KH_nn_utils.fc_layer(layer_activations["layer" + str(self.n_hidden_layers)], input_size =int(layer_activations["layer" + str(self.n_hidden_layers)].get_shape()[1]), output_size = self.action_dim, name = "FC" + str(self.n_hidden_layers+1), nonlinear = "tanh")
        action_out = tf.multiply(raw_action_output,self.torque_limit) # scaling [-1, 1] output from tanh to the available torque -> [-torque_limit, torque_limit]
        return current_state, action_out
    
    # Runs the current policy and yields the deterministic action
    def run_policy(self, state):
        return self.sess.run(self.action_out, feed_dict={self.current_state:state})
    
    # Train the policy one step
    def train(self, states, action_gradients):
        self.sess.run(self.train_actor_one_step, feed_dict={self.current_state: states, self.action_Q_gradients: action_gradients})
        
    # Update the target network one step closer to the real policy parameters
    def update_target_network(self):
        self.sess.run(self.update_target_actor_network_parameters)
        
    # Ask the target what action to take (used in calculating targets for the Q network)
    def predict_target(self, state):
        return self.sess.run(self.target_action_out, feed_dict={self.target_current_state: state})
    
    # Gets the number of trainable variables from the actor
    def get_num_trainable_variables(self):
        return self.num_trainable_variables
    
    def save_policy(self, filename, iteration):
        # Saving the trained model
        tf.saved_model.simple_save(self.sess, 'TensorBoard/' + filename + '/trainedNetworks/iteration' + str(iteration), inputs = {'state':self.current_state}, outputs = {'action':self.action_out})
        return 0
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Class contains all the workings of the critic network, including building, training, and executing
class CriticNetwork():
    """
    Critic network receives the state and action and outputs the value 
    associated with this state-action pair. Action is taken from the output 
    of the actor.
    """
    # __init__ runs automatically the first time CriticNetwork() is called
    def __init__(self, sess, n_hidden_layers, n_neurons, state_dim, action_dim, learning_rate, target_update_rate, num_actor_variables, hidden_activation):
        self.sess = sess
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = target_update_rate
        self.hidden_activation = hidden_activation
        
        # Create the policy (actor) network
        self.current_state, self.value_out, self.chosen_actions = self.create_critic_network()
        
        # Grabbing all trainable parameters of network
        self.critic_network_parameters = tf.trainable_variables()[num_actor_variables:] # from num_actor_variables to the end (i.e., the critic parameters only)
        
        # Create the critic target network
        self.target_current_state, self.target_value_out, self.target_chosen_actions = self.create_critic_network()
        
        # Getting the critic target parameters
        self.target_critic_network_parameters = tf.trainable_variables()[(len(self.critic_network_parameters) + num_actor_variables):]
        
        # Operation to update the target policy with the real policy weights. Loops through all parameters -> new_target = tau*real + (1-tau)*target
        self.update_target_critic_network_parameters = [self.target_critic_network_parameters[i].assign(tf.multiply(self.critic_network_parameters[i],
                                                       self.tau) + tf.multiply(self.target_critic_network_parameters[i], 1. - self.tau))
                                                       for i in range(len(self.target_critic_network_parameters))]
        
        
        
        # Q-network target value (y_i)
        self.Q_value_targets = tf.placeholder(tf.float32, [None, 1])
        
        # Calculating online Q-network loss function
        self.loss = tf.nn.l2_loss(self.value_out - self.Q_value_targets)
           
        # Selecting Optimizer, specifying how to minimize the loss function
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_critic_one_step = self.optimizer.minimize(self.loss)
        
        # Calculating the gradient of the Q-function with respect to the action
        # for use in the policy network training
        self.dQ_dAction = tf.gradients(self.value_out, self.chosen_actions)
        
    # Creates the policy network and returns the state placeholder and output actions
    def create_critic_network(self):
        current_state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name = "current_state") # placeholder for input to policy NN
        chosen_actions = tf.placeholder(tf.float32, shape=[None, self.action_dim], name = "chosen_action") # placeholder for action taken, which is an input to the Q function
                
        # Building n_hidden_layer critic network, with the actions introduced into the second layer       
        layer_activations = {} # initializing the dictionary that will hold the outputs of each layer
        
        # Running state through first layer
        _, layer_activations["critic_layer1"] = KH_nn_utils.fc_layer(current_state, input_size = self.state_dim, output_size = self.n_neurons, name = "FC1", nonlinear = self.hidden_activation, batch_norm = True)
        
        # Adding chosen_actions into the first layer by concatenating it with the activations of the first layer
        layer_activations["critic_layer1"] = tf.concat([layer_activations["critic_layer1"], chosen_actions], axis = 1)
        
        # Now, we can run this through the remaining layers of the critic network, as per normal (but starting at layer 2)        
        for i in (range(1,self.n_hidden_layers)): # for the second to the second last layer
            _, layer_activations["critic_layer" + str(i+1)] = KH_nn_utils.fc_layer(layer_activations["critic_layer" + str(i)], input_size =int(layer_activations["critic_layer" + str(i)].get_shape()[1]), output_size = self.n_neurons, name = "FC" + str(i+1), nonlinear = self.hidden_activation, batch_norm = True)
            
        # for the final layer, which has an output of the size of the actions and is constrained
        value_out, _ = KH_nn_utils.fc_layer(layer_activations["critic_layer" + str(self.n_hidden_layers)], input_size =int(layer_activations["critic_layer" + str(self.n_hidden_layers)].get_shape()[1]), output_size = 1, name = "FC" + str(self.n_hidden_layers+1), nonlinear = "relu")
        return current_state, value_out, chosen_actions
    
    # Runs the current policy and yields the deterministic action
    def run_critic(self, state, actions):
        return self.sess.run(self.value_out, feed_dict={self.current_state:state, self.chosen_actions:actions})
    
    # Train the policy one step
    def train(self, states, actions, target_q_values):
        return self.sess.run([self.value_out, self.train_critic_one_step], feed_dict={self.current_state: states, self.chosen_actions: actions, self.Q_value_targets:target_q_values})
                
    # Update the target network one step closer to the real policy parameters
    def update_target_network(self):
        self.sess.run(self.update_target_critic_network_parameters)
        
    # Ask the target what action to take (used in calculating targets for the Q network)
    def predict_target(self, state, actions):
        return self.sess.run(self.target_value_out, feed_dict={self.target_current_state: state, self.target_chosen_actions: actions})
    
    # Compute the gradients of the Q function with respect to the actions
    def action_gradients(self, state, actions):
        return self.sess.run(self.dQ_dAction, feed_dict={self.current_state:state, self.chosen_actions:actions})
    
    
    
    
##### Possibly remove this and use pure random exploration #####
    
# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)