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


timesteps = 100
animate_motion = 1
standard_height = 1.8
standard_mass = 75
beta_proximal = 
beta_distal = 0.43
mass_distal = 0.016*standard_mass
mass_proximal = 0.028*standard_mass
l_proximal = 0.189*standard_height
l_distal = 0.145*standard_height
inerita_distal = mass_distal*(0.303*l_distal)**2
inertia_proximal = mass_proximal*(0.368*l_proximal)**2



env = Environment() # generates & initializes dynamics environment
state = env.initial_condition() # getting the initial conditions from the environment for this trial

state_log = []

for t in range(timesteps):
        # Getting the action by running the deterministic policy with full noise
    action = 0
        
    # Step the environment
    state, reward, terminate = env.one_timestep(action.T) 
    
    state_log.append(state)
    # if we're at the last timestep
    if t == (timesteps-1):
        break
     
            
    if animate_motion:
        state_log = np.asarray(state_log)
        print("Animating...")    
        # Calculating Elbow & Wrist Positions
        x1 = length*np.cos(state_log[:,0])
        y1 = length*np.sin(state_log[:,0])            
        x2 = length*np.cos(state_log[:,0] + state_log[:,1]) + x1
        y2 = length*np.sin(state_log[:,0] + state_log[:,1]) + y1
        
        fig = plt.figure(1)
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
