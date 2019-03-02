
"""
This script provides the environment for the robotic arm simulation.

A two-link robotic arm is tasked with placing its end-effector on a specific 
location. The state contains the angle and angular rates of both links, as well
as the location of the desired end-effector location.

The reward is 1 unit per second while the end-effector is perfectly on the target,
and decays exponentially through space.

Both links have a motor that applies a continuous torque, with limits.

All dynamic environments I create will have a standardized architecture. The 
reason for this is I have one learning algorithm and many environments. All 
environments are responsible for:
    - dynamics propagation (via the step method)
    - initial conditions   (via the reset method)
    - reporting environment properties (defined in __init__)
    - seeding the dynamics (via the seed method)
    - animating the motion (via the render method):
        - Rendering is done all in one shot by passing the completed states
          from a trial to the render() method.
          
Outputs:
    Reward must be of shape ()
    State must be of shape (state_size,)
    Done must be a bool

Inputs:
    Action input is of shape (action_size,)
    
Communication with agent:
    The agent communicates to the environment through two queues:
        agent_to_env: the agent passes actions or reset signals to the environment
        env_to_agent: the environment returns information to the agent    


@author: Kirk Hovell (khovell@gmail.com)
"""
import numpy as np
import multiprocessing
from scipy.integrate import odeint # Numerical integrator

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Environment:
    
    def __init__(self): 
        ##################################
        ##### Environment Properties #####
        ##################################
        self.state_size = 6
        self.action_size = 2
        self.lower_action_bound = np.array([-0.1, -0.1]) # [Nm]
        self.upper_action_bound = np.array([ 0.1,  0.1]) # [Nm]
        self.lower_state_bound =  np.array([1., 1., 1.])
        self.upper_state_bound =  np.array([1., 1., 1.])
        self.timestep = 0.1 # [s]
        self.mass = 10. # [kg]
        self.length = 1. # [m]
        self.target_reward = 1.
        self.torque_reward = 0.
        self.ang_rate_reward = 0.
        self.reward_circle_radius = 0.05 # [m]
        self.shape_rewards = True
        self.max_ang_rate = 2000. # [rad/s]
        self.friction = 0. # coefficient of friction in joints
        self.num_frames = 100 # total animation is cut into this many frames
        self.randomize = False # whether or not to randomize the state & target location
    
    
    
    
    ###################################
    ##### Seeding the environment #####
    ###################################
    def seed(self, seed):
        np.random.seed(seed)        
        
    ######################################
    ##### Resettings the Environment #####
    ######################################    
    def reset(self):
        # This method resets the state and returns it
        
        # If we are randomizing the initial consitions and state
        if self.randomize: 
            # Randomizing initial angles and angular rates 
            self.state = np.array([np.random.rand(1)[0]*2*np.pi, np.random.rand(1)[0]*2*np.pi, np.random.rand(1)[0]*0.05, np.random.rand(1)[0]*0.05])            
            # Randomizing target location radius and angle
            desired_r = np.random.rand(1)[0]*2*self.length
            desired_angle = np.random.rand(1)[0]*2*np.pi            
        else:
            # Consistent initial conditions
            self.state = np.array([np.pi/2, -np.pi/2, 0., 0.])
            desired_r = 1.
            desired_angle = np.pi
        
        # Calculating the desired end-effector position
        self.desired_ee_position = np.array([desired_r*np.cos(desired_angle), desired_r*np.sin(desired_angle)])
        
        # Resetting the time
        self.time = 0.         
        
        # Return the state
        return np.append(self.state, self.ee_position_error())
        
    
    #####################################
    ##### Step the Dynamics forward #####
    #####################################
    def step(self, action):
        
        action = action.reshape(2,1)
        
        done = False        
        parameters = [self.length, self.mass, action, self.friction]        

        # Integrating forward one time step. 
        # Returns initial condition on first row then next timestep on the next row
        ##############################
        ##### PROPAGATE DYNAMICS #####
        ##############################
        next_states = odeint(equations_of_motion,self.state,[self.time, self.time + self.timestep], args = (parameters,), full_output = 0)
        
        # Calculating the reward for this state
        reward = self.reward_function(action) 
        
        self.state = next_states[1,:] # remembering the current state
        self.time += self.timestep # updating the stored time       
        
        # If the arm is spinning too fast, end the episode
        if (np.abs(self.state[2]) > self.max_ang_rate) | (np.abs(self.state[2] + self.state[3]) > self.max_ang_rate):
            print('Too fast! Terminating early')
            done = True

        # Return the (state, reward, done)
        return np.append(self.state, self.ee_position_error()), reward, done

    
    def ee_position_error(self):
        current_ee_position = np.array([self.length*np.cos(self.state[0]) + self.length*np.cos(self.state[0] + self.state[1]), self.length*np.sin(self.state[0]) + self.length*np.sin(self.state[0] + self.state[1])])

        return current_ee_position - self.desired_ee_position
    
    def reward_function(self, action):
        # Returns the reward for this timestep as a function of the state and action
        
        # Initializing the reward
        reward = 0.
        
        # Giving rewards for being near the target location
        current_ee_position = np.array([self.length*np.cos(self.state[0]) + self.length*np.cos(self.state[0] + self.state[1]), self.length*np.sin(self.state[0]) + self.length*np.sin(self.state[0] + self.state[1])])
        
        # If we want to shape the rewards
        if self.shape_rewards: 
            reward += np.exp(-np.linalg.norm(self.desired_ee_position - current_ee_position)) * self.target_reward
                
        else: # Sparse rewards                            
            if np.linalg.norm(self.desired_ee_position - current_ee_position) < self.reward_circle_radius:
                reward += self.target_reward
        
        # Giving penalties for using joint torque
        reward += np.abs(action[0])*self.torque_reward + np.abs(action[1])*self.torque_reward
        
        # Giving penalties for having high angular rates
        reward += np.abs(self.state[2])*self.ang_rate_reward + np.abs(self.state[2] + self.state[3]) * self.ang_rate_reward
            
        # Multiplying the reward by the timestep to give the rewards on a per-second basis
        return (reward*self.timestep).squeeze()


    def generate_queue(self):
        # Generate the queues responsible for communicating with the agent
        self.agent_to_env = multiprocessing.Queue(maxsize = 1)  
        self.env_to_agent = multiprocessing.Queue(maxsize = 1)  
        return self.agent_to_env, self.env_to_agent
        
        
    def run(self):
        ###################################
        ##### Running the environment #####
        ###################################
        """
        This method is called when the environment process is launched by main.py.
        It is responsible for continually listening for an input action from the
        agent through a Queue. If an action is received, it is to step the environment
        and return the results.
        """
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
        
#####################################################################
##### Generating differential equations representing the motion #####
#####################################################################
def equations_of_motion(state, t, parameters):
    # From the state, it returns the first derivative of the state
        
    # Unpacking parameters
    l, m, action, friction = parameters 
    
    # Unpacking state (theta1, theta2, theta1_dot, theta2_dot)
    x1, x2, x3, x4 = state

    # Mass matrix
    M = np.matrix([[5*m*l**2/3 + m*l**2*np.cos(x2), m*l**2/3 + m*l**2*np.cos(x2)/2],[m*l**2/3 + m*l**2*np.cos(x2)/2, m*l**2/3]])
    
    # C matrix
    C = np.matrix([[-m*l**2*np.sin(x2)*x4*(2*x3+x4)],[m*l**2*x3**2*np.sin(x2)/2]])
    
    # Calculating angular rate derivatives
    x3dx4d = np.array(np.linalg.inv(M)*(action.reshape(2,1) - C - friction*np.matrix([[x3],[x4]]))).squeeze() 

    # Building the derivative matrix d[x1, x2, x3, x4]/dt = [x3, x4, x3dx4d]
    derivatives = np.concatenate((np.array([x3,x4]),x3dx4d)) 

    return derivatives


##########################################
##### Function to animate the motion #####
##########################################
def render(states, actions, episode_number, filename):
    print("Animating...")    
    
    # Calculating Elbow & Wrist Positions throughout entire animation
    elbow_x = self.length*np.cos(states[:,0])
    elbow_y = self.length*np.sin(states[:,0])            
    wrist_x = self.length*np.cos(states[:,0] + states[:,1]) + elbow_x
    wrist_y = self.length*np.sin(states[:,0] + states[:,1]) + elbow_y
    
    # Generating figure window
    fig = plt.figure(episode_number)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    fig.set_size_inches(5,5,True)
    ax.grid()
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    
    # Defining plot properties        
    line, = ax.plot([], [], 'o-', lw=2)
    scat = ax.scatter([], [], c='r')
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    # Function called once to initialize axes as empty
    def initialize_axes():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text, scat
    
    # Function called repeatedly to draw each frame
    def render_one_frame(frame):
        # Combining the X and Y positions to draw the arm 
        thisx = [0, elbow_x[frame], wrist_x[frame]]
        thisy = [0, elbow_y[frame], wrist_y[frame]]                   
        # Draw the arm
        line.set_data(thisx, thisy)            
        # Draw the end-effector position
        scat.set_offsets(self.desired_ee_position.reshape(1,2))            
        # Draw the title
        time_text.set_text(time_template%(frame*self.timestep))
        
        # Since blit = True, must return everything that has changed at this frame
        return line, time_text, scat
    
    # Generate the animation!
    ani = animation.FuncAnimation(fig, render_one_frame, frames = np.linspace(1, len(states)-1,self.num_frames).astype(int),
                                  blit = True, init_func = initialize_axes, fargs = None, repeat_delay = 1000)
    """
    frames = the int that is passed to render_one_frame. I use it to selectively plot certain data
    fargs = additional arguments for render_one_frame
    interval = delay between frames in ms
    """
    
    # Save the animation!
    ani.save(filename = 'TensorBoard/' + filename + '/videos/episode_' + str(episode_number) + '.mp4', fps=30,dpi=100)
    plt.show()
    print('Done!')
    
