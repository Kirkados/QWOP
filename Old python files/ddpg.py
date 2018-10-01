"""
Kirk Hovell - April 23, 2018
PhD Comprehensive Exam

Attempts to demonstrate robustness in transfering from simulation to a real environment
where parameters may be different than used in simulation.

Here, joint friction is considered.

May need to implement D4PG to speed up learning
Implement tensorflow checkpoints so that trained networks can be loaded in.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dynamics import Environment
import time
import datetime
import tensorflow as tf
from actors_and_critics import PolicyNetwork, CriticNetwork, OrnsteinUhlenbeckActionNoise
from replay_buffer import ReplayBuffer
#import argparse


#def main(args):
tf.reset_default_graph() # Wipes tensorflow graph clean



#%% Runtime Parameters
run_name = 'sparse_rewards' # for animator
plot_trials = [10,50,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000] # which trials to plot & animate
plot_figs = 0 # 1 = plot; 0 = do not
animate_motion = 1 # 1 = animate; 0 = do not
num_frames = 100 # [in animation] total animation length played over this many frames
seed = 0 # seed
output_every_num_iterations= 1000 # how often to print status to screen
test_time_every_num_iterations = 20 # how often to test the policy for Tensorboard
test_time_trials = 1 # how many trials to include in the policy testing
test_time_animate = [0, 1] # which entries of each test to animate
checkpoint_frequency = 1000 # save a checkpoint every ### iterations

# Environment parameters
simulation_length = 40 # [s]
timestep = 0.1 # [s]
target_reward = 1 # [reward/second] reward for placing end-effector on the target
torque_reward = -0 # [reward/second] penalty for using torque
ang_rate_reward = -0 # [reward/second] penalty for having a high angular rate
reward_shaping = 0 # whether or not to shape the reward towards the goal
reward_circle_radius = 0.05 # [m] distance within to receive the position reward
action_dimension = 2 # number of actions
state_dimension = 6 # number of states input to the policy
max_ang_rate = 2000 # [rad/s] maximum angular rate of any link before episode is terminated
randomize = 0 # 0 = consistent initial conditions; 1 = randomized initial conditions and target location
noise_sigma = 0.04 # Ornstein-Uhlenbeck noise parameter

# Training Parameters
num_iterations = 15001
actor_learning_rate = 0.001 # learning for policy neural network
critic_learning_rate = 0.0001
target_update_rate = 0.001 # tau
gamma = 0.99 # amount of Q' network to take for targets
n_hidden_layers = 2 # number of hidden layers in the neural networks
n_neurons = 300 # number of neurons in each hidden layer
hidden_activation = 'relu' # hidden layer nonlinear activation function
batch_size = 128 # [samples] sampled from replay buffer to train actor and critic
buffer_size = 800000 # [timesteps] to be included in the replay buffer

# Arm Properties
min_length = 1. # [m]
max_length = 1. # [m]
mass = 10. # [kg]
friction = 0 # [coefficient of friction in joints]
torque_limit = 0.1 # [Nm] limits on arm actuators

#%%
# Set random seeds
tf.set_random_seed(seed)
np.random.seed(seed)

target_location_over_trials = []

timesteps = int(simulation_length/timestep)
filename = run_name + '-{:%Y-%m-%d %H-%M}'.format(datetime.datetime.now())

#%% Logging Parameters
log_parameters = {'num_iterations':num_iterations, 'simulation_length':simulation_length,
            'timestep':timestep,'plot_trials':plot_trials,'animate_motion':animate_motion,'num_frames':num_frames,
            'target_reward':target_reward,'torque_reward':torque_reward,'ang_rate_reward':ang_rate_reward,
            'reward_shaping':reward_shaping,'reward_circle_radius':reward_circle_radius,
            'actor_learning_rate':actor_learning_rate,'critic_learning_rate':critic_learning_rate,'action_dimension':action_dimension,'state_dimension':state_dimension,
            'gamma':gamma,'buffer_size':buffer_size,'batch_size':batch_size,'seed':seed,'n_neurons':n_neurons,'min_length':min_length,'max_length':max_length,'mass':mass,
            'n_hiddenlayers':n_hidden_layers,'n_neurons':n_neurons,'target_update_rate':target_update_rate,'torque_limit':torque_limit,
            'test_time_every_num_iterations':test_time_every_num_iterations,'test_time_trials':test_time_trials,'test_time_animate':test_time_animate,
            'randomize':randomize,'max_ang_rate':max_ang_rate,'noise_sigma':noise_sigma,'hidden_activation':hidden_activation}
#%% Starting the rollouts
start_time = time.time()
with tf.Session() as sess:
    
    # Generating Actor & Critic Networks
    actor = PolicyNetwork(sess, n_hidden_layers, n_neurons, torque_limit, state_dimension, action_dimension,  actor_learning_rate, batch_size, target_update_rate, hidden_activation)
    critic = CriticNetwork(sess,n_hidden_layers, n_neurons,               state_dimension, action_dimension, critic_learning_rate,             target_update_rate, actor.get_num_trainable_variables(), hidden_activation)
    
    # Actor noise function, to be modified later
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dimension), sigma = noise_sigma) # sigma = 0.05 for more_time good run
    
    #%% Tensorboard logging
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", episode_ave_max_q)
    
    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()
    
    init = tf.global_variables_initializer() 
    saver = tf.train.Saver() # Getting ready to save
    
    writer = tf.summary.FileWriter('TensorBoard/' + filename)
    
    sess.run(init) # initialize the variables
    writer.add_graph(sess.graph)
    
               
    #%% Logging parameters before things get going
    np.save('TensorBoard/' + filename + '/run_parameters.npy',log_parameters)
    
    # To load in parameters
    #filename_to_load = 'TEST-2018-03-07 17-48'
    #locals().update(np.load('TensorBoard/' + filename_to_load + '/run_parameters.npy').item())

    #%% Initializing target network weights (many times so that they start off relatively equaly.)
    for i in range(int(1/target_update_rate)):
        actor.update_target_network()
        critic.update_target_network()
    
    # Initializing replay buffer 
    replay_buffer = ReplayBuffer(buffer_size, seed)
    
    for i in range(num_iterations): # number of total training iterations
        if (i % output_every_num_iterations) == 0:
               print("*****Iteration: %i *****" %i)
               
        # Logging variables
        episode_average_max_Q = 0
        episode_reward = 0
        
        # Choosing the length for this trial
        length = np.random.uniform(low = min_length, high = max_length)
        
        env = Environment(timestep, mass, length, target_reward, torque_reward, ang_rate_reward, reward_circle_radius, reward_shaping, max_ang_rate, randomize, friction) # generates & initializes dynamics environment
        target_location_over_trials.append(env.target_location()) # getting the desired EE position for this trial
        state = env.initial_condition() # getting the initial conditions from the environment for this trial

        for t in range(timesteps):
            # Getting the action by running the deterministic policy with full noise
            action = np.clip(actor.run_policy(np.reshape(state, [-1, state_dimension])) + actor_noise(), -torque_limit, torque_limit) # clipping to ensure within actuator limits
            
            # Step the environment
            new_state, reward, terminate = env.one_timestep(action.T) 
            
            # if we're at the last timestep
            if t == (timesteps-1):
                   terminate = True
                   
            # Add to replay buffer
            replay_buffer.add(np.reshape(state, [-1, state_dimension]), np.reshape(action, [-1, action_dimension]), reward, 
                              np.reshape(new_state, [-1, state_dimension]), terminate)
            
            # Check if the replay buffer is full enough to give us a batch
            if replay_buffer.size() > batch_size:
                # Sample a batch
                state_batch, action_batch, reward_batch, next_state_batch, terminate_batch = replay_buffer.sample_batch(batch_size)
                
                # Calculate Q' values (Q'(s_{t+1}, mu'(s_{t+1})))
                Q_prime_values = critic.predict_target(next_state_batch, actor.predict_target(next_state_batch))
                
                # Calculate Q targets.
                Q_value_targets = []
                for k in range(batch_size): # for each entry in the batch
                    if terminate_batch[k]: # if this was the last timestep before termination
                        Q_value_targets.append(reward_batch[k]) # Q(s,a) = r(t) because there are no further actions
                    else:
                        Q_value_targets.append(reward_batch[k] + gamma * Q_prime_values[k]) # reward plus expected future rewards
         
                    
                # Train the critic! predicted_Q_value should equal Q_value_targets
                predicted_Q_value, _ = critic.train(state_batch, action_batch, np.reshape(Q_value_targets,[batch_size, 1]))
                
                # Adding the maximum predicted_Q_value from the online critic from training at this timestep
                episode_average_max_Q += np.amax(predicted_Q_value)
                
                # Train the actor
                action_outs = actor.run_policy(state_batch) # get actions taken from this batch of states. No noise compared to action_batch
                action_gradients = critic.action_gradients(state_batch, action_outs) # gradients of Q with respect to the actions
                actor.train(state_batch, action_gradients[0]) # train the actor one step
                
                # Update the target networks a little bit closer to the online ones
                actor.update_target_network()
                critic.update_target_network()
                
            
            # Setting state for next timestep
            state = new_state  
            
            # Accumulating reward achieved on this rollout
            episode_reward += reward
            
            # If we need to end early
            if terminate: 
                break
        
         
        # Test Time!!!! Evaluate the policy and see how it's doing
        if (i % test_time_every_num_iterations) == 0: # if it's test time

            episode_reward = 0
            
            for k in range(test_time_trials): # averaging over a few trials (useful if each sim is initiallized differently)
                
                state_log = []
                action_log = []
                new_state_log = []
                reward_log = []
                
                # Choosing the length for this trial
                length = np.random.uniform(low = min_length, high = max_length)
                
                testEnv = Environment(timestep, mass, length, target_reward, torque_reward, ang_rate_reward, reward_circle_radius, reward_shaping, max_ang_rate, randomize, friction) # generates & initializes dynamics environment
                
                target_location = testEnv.target_location() # getting the desired EE position for this trial
                state = testEnv.initial_condition() # getting the initial conditions from the environment for this trial
                

 
                for t in range(timesteps):              
                    # Run the deterministic policy to get the action :)
                    action = actor.run_policy(np.reshape(state, [-1, state_dimension]))
                    
                    # Step the environment
                    new_state, reward, terminate = testEnv.one_timestep(action.T) 
                    
                    # Accumulate reward
                    episode_reward += reward
                    
                    # Logging data to plot/animate
                    action_log.append(np.squeeze(action))
                    state_log.append(state)
                    new_state_log.append(new_state)
                    reward_log.append(reward)
                    
                    state = new_state
                    
                    if terminate:
                           break
                
                 
                if animate_motion & np.any([x == k for x in test_time_animate]) & np.any([p == i for p in plot_trials]):  
                    state_log = np.asarray(state_log)
                    print("Animating...")    
                    # Calculating Elbow & Wrist Positions
                    x1 = length*np.cos(state_log[:,0])
                    y1 = length*np.sin(state_log[:,0])            
                    x2 = length*np.cos(state_log[:,0] + state_log[:,1]) + x1
                    y2 = length*np.sin(state_log[:,0] + state_log[:,1]) + y1
                    
                    fig = plt.figure(i+k)
                    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
                    fig.set_size_inches(5,5,True)
                    ax.grid()
                    plt.xlabel('X Position (m)')
                    plt.ylabel('Y Position (m)')
                    
                    line, = ax.plot([], [], 'o-', lw=2)
                    scat = ax.scatter([], [], c='r')
                    time_template = 'time = %.1fs'
                    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
                    
                    def init(): # initializes aces
                        line.set_data([], [])
                        time_text.set_text('')
                        return line, time_text, scat
                    
                    def animate(j): # draws current frame
                        thisx = [0, x1[j], x2[j]]
                        thisy = [0, y1[j], y2[j]]         
                        line.set_data(thisx, thisy)
                        scat.set_offsets(target_location.reshape(1,2))
                        time_text.set_text(time_template%(j*timestep))
                        return line, time_text, scat
                    
                    ani = animation.FuncAnimation(fig, animate, np.linspace(1, len(state_log)-1,num_frames).astype(int),
                                                  interval=25, blit=True, init_func=init)
                    
                    ani.save('TensorBoard/' + filename + '/trial_' + str(i) + '_' + str(k) + '.mp4', fps=30,dpi=100)
                    plt.show()
                    print('Done!')
                
                
            # all test trials are done, find the average reward and log it
            average_reward = episode_reward/float(test_time_trials)                    
            summary_string = sess.run(summary_ops, feed_dict={summary_vars[0]: average_reward[0], summary_vars[1]: episode_average_max_Q / float(timesteps)})                       
            writer.add_summary(summary_string, i)
            writer.flush()
        
        if (i % checkpoint_frequency) == 0:
            print('Saving Checkpoint ' + str(i))
            saver.save(sess, 'TensorBoard/' + filename + '/checkpoints/iteration_' + str(i) + '.ckpt')
            
            # Saving the trained policy    
            actor.save_policy(filename, i)   
        
end_time = time.time()
print("Execution took %f seconds" %(end_time - start_time))
