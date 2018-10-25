"""
Builds the tensorflow graph neural networks for the actor and critic
"""

import tensorflow as tf

from settings import Settings

def build_actor_network(state, trainable, scope):
    """ 
    Build the actor network that receives the state and outputs the action.
    
    Inputs:
        state:      A placeholder where the input will be fed when used.
        trainable:  Whether the network can be trained (learner) or if its weights are frozen (actors)
        scope:      Name of the tensorflow scope
    """   
    
    # Making sure all variables generated here are under the name "scope"
    with tf.variable_scope(scope):
        
        layer = state
        
        # If learning from pixels
        if Settings.LEARN_FROM_PIXELS:
            
            # Build convolutional layers
            for i, conv_layer_settings in enumerate(Settings.CONVOLUTIONAL_LAYERS):
                layer = tf.layers.conv2d(inputs = layer,
                                         activation = tf.nn.relu,
                                         trainable = trainable,
                                         name = 'conv_layer' + str(i),
                                         **conv_layer_settings) # each layer is named layer
            
            layer = tf.layers.flatten(layer) # flattening image into a column for subsequent layers
        
        
        # Building fully connected layers
        for i, number_of_neurons in enumerate(Settings.ACTOR_HIDDEN_LAYERS):
            layer = tf.layers.dense(inputs = layer,
                                    units = number_of_neurons,
                                    trainable = trainable,
                                    activation = tf.nn.relu,
                                    name = 'fully_connected_layer_' + str(i))
        
        # Convolutional layers (optional) have been applied, followed by fully-connected hidden layers
        # The final layer goes from the output of the last hidden layer
        # to the action size. It is squished with a sigmoid and then scaled to the action range
        actions_out_unscaled = tf.layers.dense(inputs = layer,
                                      units = Settings.ACTION_SIZE,
                                      trainable = trainable,
                                      activation = tf.nn.sigmoid,
                                      name = 'output_layer') # sigmoid forces output between 0 and 1... need to scale it to the action range
        
        # Scaling actions to the correct range
        action_scaled = Settings.LOWER_ACTION_BOUND + tf.multiply(actions_out_unscaled, Settings.ACTION_RANGE)
        
        return action_scaled
        
        
def build_Q_network(state, action, trainable, reuse, scope):
    """
    Defines a critic network that predicts the Q-value (expected future return)
    from a given state and action. 
    """
    with tf.variable_scope(scope):
        
        layer = tf.concat([state, action], axis = 1) # concatenating state and action for input
        
        # If learning from pixels
        if Settings.LEARN_FROM_PIXELS:
            
            # Build convolutional layers
            for i, conv_layer_settings in enumerate(Settings.CONVOLUTIONAL_LAYERS):
                layer = tf.layers.conv2d(inputs = layer,
                                         activation = tf.nn.relu,
                                         trainable = trainable,
                                         name = 'conv_layer' + str(i),
                                         **conv_layer_settings) # each layer is named layer
            
            layer = tf.layers.flatten(layer) # flattening image into a column for subsequent layers 
        
        
        # Building fully connected layers
        for i, number_of_neurons in enumerate(Settings.CRITIC_HIDDEN_LAYERS):
            layer = tf.layers.dense(inputs = layer,
                                    units = number_of_neurons,
                                    trainable = trainable,
                                    activation = tf.nn.relu,
                                    name = 'fully_connected_layer_' + str(i))
        
        # Convolutional layers (optional) have been applied, followed by fully-connected hidden layers
        # The final layer goes from the output of the last hidden layer
        # to the distribution size (NUMBER_OF_ATOMS).
        q_value = tf.layers.dense(inputs = layer,
                                      units = Settings.NUMBER_OF_ATOMS,
                                      trainable = trainable,
                                      reuse = reuse,
                                      activation = tf.nn.softmax,
                                      name = 'output_layer') # softmax ensures that all outputs add up to 1, relative to their weights
 
        return q_value