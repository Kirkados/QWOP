"""
These two Classes build the Actor network and Critic network.
The actor network receives the state and calculates the action.
The critic network receives the state and action and calculates a q-distribution.

The q-distribution represents the probability that the value of this state-action pair
is in a certain bin. Each of the outputs corresponds to a probability that the true
value lies in a given bin. This strategy yields better results than simply estimating
the value of the state-action pair, as we have a full distribution to work with rather
than just the mean.

@author: Kirk Hovell (khovell@gmail.com)
"""

import tensorflow as tf

from settings import Settings


class BuildQNetwork:
    
    def __init__(self, state, scope):
        
        self.state = state
        self.scope = scope
        """
        Defines a critic network that predicts the q-distribution (expected return)
        from a given state and action. 
        
        The network archetectire is modified from the D4PG paper. The state goes through
        two layers on its own before being added to the action who has went through
        one layer. Then, the sum of the two goes through the final layer. Note: the 
        addition happend before the relu.
        """
        with tf.variable_scope(self.scope):
            # Two sides flow through the network independently.
            self.layer  = self.state
            
            ###########################################
            ##### (Optional) Convolutional Layers #####
            ###########################################
            # If learning from pixels (a state-only feature), use convolutional layers
            if Settings.LEARN_FROM_PIXELS:            
                # Build convolutional layers
                for i, conv_layer_settings in enumerate(Settings.CONVOLUTIONAL_LAYERS):
                    self.layer = tf.layers.conv2d(inputs = self.layer,
                                                       activation = tf.nn.relu,
                                                       name = 'state_conv_layer' + str(i),
                                                       **conv_layer_settings) # the "**" allows the passing of keyworded arguments
                
                # Flattening image into a column for subsequent layers 
                self.layer = tf.layers.flatten(self.layer) 
                    
            ##################################
            ##### Fully connected layers #####
            ##################################
            for i, number_of_neurons in enumerate(Settings.CRITIC_HIDDEN_LAYERS):
                self.layer = tf.layers.dense(inputs = self.layer,
                                                  units = number_of_neurons,
                                                  activation = tf.nn.relu,
                                                  name = 'state_fully_connected_layer_' + str(i))            
           
            #################################################
            ##### Final Layer to get Value Distribution #####
            #################################################
            # Calculating the final layer logits as an intermediate step,
            # since the cross_entropy loss function needs logits.
            q_distribution_logits = []
            q_distribution = []
            for i in range(Settings.ACTION_SIZE):
                
                q_logits = tf.layers.dense(inputs = self.layer,
                                           units = Settings.NUMBER_OF_BINS,
                                           activation = None,
                                           name = 'output_layer_' + str(i))
                
                # Appending the output distribution for each possible action
                q_distribution_logits.append(q_logits) # logits
                q_distribution.append(tf.nn.softmax(q_logits)) # probabilities
            
            # Assembling output into [batch_size, # actions, # bins]
            self.q_distribution_logits = tf.stack(q_distribution_logits, axis = 1)
            self.q_distribution = tf.stack(q_distribution, axis = 1)
            
            # The value bins that each probability corresponds to.
            self.bins = tf.lin_space(Settings.MIN_Q, Settings.MAX_Q, Settings.NUMBER_OF_BINS)
            
            # Getting the parameters from the critic
            self.parameters = tf.trainable_variables(scope=self.scope)            
            
    def generate_training_function(self, action_placeholder, target_q_distribution, target_bins, importance_sampling_weights):
        # Create the operation that trains the critic one step.        
        with tf.variable_scope(self.scope):
            with tf.variable_scope('Training'):
                
                # Choosing the Adam optimizer to perform stochastic gradient descent
                self.optimizer = tf.train.AdamOptimizer(Settings.CRITIC_LEARNING_RATE)               
                
                # Project the target distribution onto the bounds of the original network
                projected_target_distribution = l2_project(target_bins, target_q_distribution, self.bins)  
                
                # Getting the q-distribution for just the chosen action
                chosen_action_index = tf.stack((tf.range(Settings.MINI_BATCH_SIZE), action_placeholder), axis = 1)
                self.q_distribution_logits_for_chosen_action = tf.gather_nd(self.q_distribution_logits, chosen_action_index) # [batch_size, # bins]
                
                # This ensures we only train the appropraite distribution (logits) from the q-network that correspondd to the chosen action                
                
                # Calculate the cross entropy loss between the projected distribution and the main q_network!
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.q_distribution_logits_for_chosen_action, labels = tf.stop_gradient(projected_target_distribution))
                
                # A loss correction is needed if we use a prioritized replay buffer
                # to account for the bias introduced by the prioritized sampling.
                if Settings.PRIORITY_REPLAY_BUFFER:
                    # Correct prioritized loss bias using importance sampling
                    self.weighted_loss = self.loss * importance_sampling_weights
                else:
                    self.weighted_loss = self.loss

                # Taking the average across the batch
                self.mean_loss = tf.reduce_mean(self.weighted_loss)
                
                # Optionally perform L2 regularization, where the network is 
                # penalized for having large parameters
                if Settings.L2_REGULARIZATION:
                    self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.parameters if 'kernel' in v.name]) * Settings.L2_REG_PARAMETER
                else:
                    self.l2_reg_loss = 0.0
                    
                # Add up the final loss function
                self.total_loss = self.mean_loss + self.l2_reg_loss
                 
                # Set the optimizer to minimize the total loss, and do so by modifying the critic parameter.
                critic_training_function = self.optimizer.minimize(self.total_loss, var_list=self.parameters)
                  
                return critic_training_function


# Projection function used by the critic training function
'''
## l2_projection ##
# Taken from: https://github.com/deepmind/trfl/blob/master/trfl/dist_value_ops.py
# Projects the target distribution onto the support of the original network [Vmin, Vmax]
'''

def l2_project(z_p, p, z_q):
    """Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.
    The supports z_p and z_q are specified as tensors of distinct atoms (given
    in ascending order).
    Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
    support z_q, in particular Kq need not be equal to Kp.
    Args:
      z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
      p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
      z_q: Tensor holding support to project onto, shape `[Kq]`.
    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """
    # Broadcasting of tensors is used extensively in the code below. To avoid
    # accidental broadcasting along unintended dimensions, tensors are defensively
    # reshaped to have equal number of dimensions (3) throughout and intended
    # shapes are indicated alongside tensor definitions. To reduce verbosity,
    # extra dimensions of size 1 are inserted by indexing with `None` instead of
    # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
    # `[k, l]' to one of shape `[k, 1, l]`).
    
    # Extract vmin and vmax and construct helper tensors from z_q
    vmin, vmax = z_q[0], z_q[-1]
    d_pos = tf.concat([z_q, vmin[None]], 0)[1:]  # 1 x Kq x 1
    d_neg = tf.concat([vmax[None], z_q], 0)[:-1]  # 1 x Kq x 1
    # Clip z_p to be in new support range (vmin, vmax).
    z_p = tf.clip_by_value(z_p, vmin, vmax)[:, None, :]  # B x 1 x Kp
    
    # Get the distance between atom values in support.
    d_pos = (d_pos - z_q)[None, :, None]  # z_q[i+1] - z_q[i]. 1 x B x 1
    d_neg = (z_q - d_neg)[None, :, None]  # z_q[i] - z_q[i-1]. 1 x B x 1
    z_q = z_q[None, :, None]  # 1 x Kq x 1
    
    # Ensure that we do not divide by zero, in case of atoms of identical value.
    d_neg = tf.where(d_neg > 0, 1./d_neg, tf.zeros_like(d_neg))  # 1 x Kq x 1
    d_pos = tf.where(d_pos > 0, 1./d_pos, tf.zeros_like(d_pos))  # 1 x Kq x 1
    
    delta_qp = z_p - z_q   # clip(z_p)[j] - z_q[i]. B x Kq x Kp
    d_sign = tf.cast(delta_qp >= 0., dtype=p.dtype)  # B x Kq x Kp
    
    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
    # Shape  B x Kq x Kp.
    delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]  # B x 1 x Kp.
    return tf.reduce_sum(tf.clip_by_value(1. - delta_hat, 0., 1.) * p, 2)