"""
Created April 24, 2018

This script loads in a pre-trained neural network and uses it as a controller 
for a dynamical system

Can test how a trained neural network works when "deployed"

@author: Kirk
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dynamics import Environment
import time
import tensorflow as tf


tf.reset_default_graph()
filename =  'length_robust-2018-05-24 15-44' # location of learned neural network model
iteration = 4000 # iterations
second_filename = 'robust_L20'

#%% Runtime Parameters
animate_motion = 1 # 1 = animate; 0 = do not
num_frames = 100 # [in animation] total animation length played over this many frames
seed = 0 # seed

# Environment parameters
num_iterations = 1
simulation_length = 40 # [s]
timestep = 0.1 # [s]
target_reward = 1 # [reward/second] reward for placing end-effector on the target
torque_reward = -0 # [reward/second] penalty for using torque
ang_rate_reward = -0 # [reward/second] penalty for having a high angular rate
reward_shaping = 1 # whether or not to shape the reward towards the goal
reward_circle_radius = 0.05 # [m] distance within to receive the position reward
max_ang_rate = 500000 # [rad/s] maximum angular rate of any link before episode is terminated
randomize = 0 # 0 = consistent initial conditions; 1 = randomized initial conditions and target location
state_dimension = 6
action_dimension = 2

# Arm Properties
length = 2. # [m]
mass = 10. # [kg]
joint_friction = 0. # [friction coefficient]
torque_limit = 0.1 # [Nm] limits on arm actuators

#%%
# Set random seeds
tf.set_random_seed(seed)
np.random.seed(seed)

target_location_over_trials = []

timesteps = int(simulation_length/timestep)
# Logging variables
episode_reward = 0

#%% Starting the rollouts
start_time = time.time()
with tf.Session() as sess:
    tf.saved_model.loader.load(sess, ['serve'], export_dir='TensorBoard/' + filename + '/trainedNetworks/iteration' + str(iteration))

    for i in range(num_iterations): # number of total training iterations
        print("*****Iteration: %i *****" %i)
        state_log = []              
                       
        env = Environment(timestep, mass, length, target_reward, torque_reward, ang_rate_reward, reward_circle_radius, reward_shaping, max_ang_rate, randomize, joint_friction) # generates & initializes dynamics environment
        target_location = env.target_location()
        state = env.initial_condition() # getting the initial conditions from the environment for this trial

        for t in range(timesteps):
            # Getting the action by running the deterministic policy with full noise
            action = sess.run('Mul:0', feed_dict={'current_state:0':np.reshape(state, [-1, state_dimension])}) # clipping to ensure within actuator limits
            
            # Step the environment
            new_state, reward, terminate = env.one_timestep(action.T) 
            
            # Accumulate reward
            episode_reward += reward
            
            # Logging data to plot/animate
            state_log.append(state)
            
            state = new_state

        if animate_motion:  
            state_log = np.asarray(state_log)
            print("Animating...")    
            # Calculating Elbow & Wrist Positions
            x1 = length*np.cos(state_log[:,0])
            y1 = length*np.sin(state_log[:,0])            
            x2 = length*np.cos(state_log[:,0] + state_log[:,1]) + x1
            y2 = length*np.sin(state_log[:,0] + state_log[:,1]) + y1
            
            fig = plt.figure(i)
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
            
            ani.save('TensorBoard/' + filename + '/deploy_' + second_filename + '_' + str(i) + '.mp4', fps=30,dpi=100)
            plt.show()
            print('Done!')
                
    # Trials are complete, finding the average reward
    print('The average reward was %f' %(episode_reward/num_iterations))
        
    
end_time = time.time()
print("Execution took %f seconds" %(end_time - start_time))