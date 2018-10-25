"""
This script contains certain functions useful to neural networks.

It includes:
    get_vars:  grabs all the variables with a given scope (e.g., all the trainable parameters from 'agent_4' for example)
    copy_vars: copies all the variables from a source to a destination, used to update the agent parameters with the most recent ones from the learner.
"""

import tensorflow as tf

def get_variables(agent_name, trainable):
    # Gets every tensorflow variable from a defined scope
    
    if trainable:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = agent_name) # grab only trainable variables
    else:
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,    scope = agent_name) # grab non-trainable variables




def copy_variables(source_variables, destination_variables, tau):
    # Copy every variable from the source to the corresponding variable in the destination
    # tau is the update rate (less than 1 for target networks)
        
    for source_variable, destination_variable in zip(source_variables, destination_variables):
        
        # Check if the source and destination variables represent the same thing (so we copy correctly)
        source_name, destination_name = source_variable.name, destination_variable.name
        assert source_name[source_name.find("/"):] == destination_name[destination_name.find("/"):] # will throw an error if not all variable names line up
        
        ###################################################################
        #### Assigning variables from source to destination one-by-one ####
        ###################################################################
        destination_variable.assign(tau * source_variable + (1 - tau) * destination_variable)
            
        
def l2_regularization(parameters):
    # For a given set of parameters, calculate the sum of the square of all the weights (ignore the biases)
    # This is added to the critic loss function to penalize it for having large weights
    running_total = 0
    
    for each_parameter in parameters:
        
        if not 'bias' in each_parameter.name: # if not a bias
            
            running_total += 1e-6 * tf.nn.l2_loss(each_parameter)
    
    
    return running_total
    