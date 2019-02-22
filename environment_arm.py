
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
    


@author: Kirk Hovell (khovell@gmail.com)
"""
import numpy as np
import math
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
        print(parameters)

        # Integrating forward one time step. 
        # Returns initial condition on first row then next timestep on the next row
        ##############################
        ##### PROPAGATE DYNAMICS #####
        ##############################
        next_states = odeint(equations_of_motion,self.state,[self.time, self.time + self.timestep], args = (parameters,), full_output = 0)
        
        reward = self.reward_function(action) 
        
        self.state = next_states[1,:] # remembering the current state
        self.time += self.timestep # updating the stored time
        
       
        
        # If the arm is spinning too fast, end the episode
        if (np.abs(self.state[2]) > self.max_ang_rate) | (np.abs(self.state[2] + self.state[3]) > self.max_ang_rate):
            print('Too fast! Terminating early')
            done = True

        # Return the (state, reward, done)
        return np.append(self.state, self.ee_position_error()), reward, done

    
    def seed(self, seed = None):
        # This method seeds the environment, for reproducability
        np.random.seed(seed)
    
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

    
    def render(self, states, episode_number, filename):
        states = np.asarray(states)
        print("Animating...")    
        # Calculating Elbow & Wrist Positions
        x1 = self.length*np.cos(states[:,0])
        y1 = self.length*np.sin(states[:,0])            
        x2 = self.length*np.cos(states[:,0] + states[:,1]) + x1
        y2 = self.length*np.sin(states[:,0] + states[:,1]) + y1
        
        fig = plt.figure(episode_number)
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
            scat.set_offsets(self.desired_ee_position.reshape(1,2))
            time_text.set_text(time_template%(j*self.timestep))
            return line, time_text, scat
        
        ani = animation.FuncAnimation(fig, animate, np.linspace(1, len(states)-1,self.num_frames).astype(int),
                                      interval=25, blit=True, init_func=init)
        
        ani.save('TensorBoard/' + filename + '/videos/episode_' + str(episode_number) + '.mp4', fps=30,dpi=100)
        plt.show()
        print('Done!')
        

def equations_of_motion(state, t, parameters):
    # From the state, it returns the first derivative of the state
    
    # Unpacking the state
    x, y, theta, x1, y1, theta1, x2, y2, theta2, xdot, ydot, thetadot, x1dot, y1dot, theta1dot, x2dot, y2dot, theta2dot = state
    
    # Unpacking parameters
    m, m1, m2, eta, eta1, eta2, gamma1, gamma2, I, I1, I2, g = parameters 
    
    fF1 = 0.
    fF2 = 0.
    phi1 = np.pi/4
    phi2 = -np.pi/4
    fN1 = 0.
    fN2 = 0.
    K = 100.
    

    # Mass matrix
    M = np.matrix([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                    m,                   0.,                                                0.,                               m1,                               0.,                            0.,                               m2,                               0.,                            0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                   0.,                   m ,                                                0.,                               0.,                               m1,                            0.,                               0.,                               m2,                            0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., -m*eta*np.cos(theta), -m*eta*np.sin(theta),                                                 I,                               0.,                               0.,                            0.,                               0.,                               0.,                            0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                  -1.,                   0.,          eta*np.sin(theta) + gamma1*np.cos(theta),                               1.,                               0., gamma1*np.sin(theta + theta1),                               0.,                               0.,                            0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                   0.,                  -1., eta*np.cos(theta) + gamma1*np.cos(theta + theta1),                               0.,                               1., gamma1*np.cos(theta + theta1),                               0.,                               0.,                            0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                   0.,                   0.,                                                0., m1*gamma1*np.cos(theta + theta1), m1*gamma1*np.sin(theta + theta1),                            I1,                               0.,                               0.,                            0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                  -1.,                   0., eta*np.sin(theta) + gamma2*np.sin(theta + theta2),                               0.,                               0.,                            0.,                               1.,                               0., gamma2*np.sin(theta + theta2)],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                   0.,                  -1., eta*np.cos(theta) + gamma2*np.cos(theta + theta2),                               0.,                               0.,                            0.,                               0.,                               1., gamma2*np.cos(theta + theta2)],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                   0.,                   0.,                                                0.,                               0.,                               0.,                            0., m2*gamma2*np.cos(theta + theta2), m2*gamma2*np.sin(theta + theta2),                            I2]])
    
    # C matrix
    C = np.matrix([[xdot],
                   [ydot],
                   [thetadot],
                   [x1dot],
                   [y1dot],
                   [theta1dot],
                   [x2dot],
                   [y2dot],
                   [theta2dot],
                   [fF1 + fF2],
                   [fN1 + fN2 - (m + m1 + m2)*g],
                   [-K*(phi1 - theta1 + phi2 - theta2) + m*g*np.sin(theta)],
                   [-thetadot**2*eta*np.cos(theta) - (thetadot + theta1dot)**2*gamma1*np.cos(theta + theta1)],
                   [ thetadot**2*eta*np.sin(theta) + (thetadot + theta1dot)**2*gamma1*np.sin(theta + theta1)],
                   [-g*gamma1*np.sin(theta+theta1) + fN1*(eta1*np.sin(theta+theta1) + gamma1*np.sin(theta+theta1)) + fF1*(eta1*np.cos(theta+theta1) + gamma1*np.cos(theta + theta1)) + k*(phi1 - theta1)],
                   [-thetadot**2*eta*np.cos(theta) - (thetadot + theta2dot)**2*gamma2*np.cos(theta+theta2)],
                   [thetadot**2*eta*np.sin(theta) + (thetadot + theta2dot)**2*gamma2*np.sin(theta + theta2)],
                   [-g*gamma2*np.sin(theta + theta2) + fN2*(eta2*np.sin(theta + theta2) + gamma2*np.sin(theta + theta2)) + fF2*(eta2*np.cos(theta + theta2) + gamma2*np.cos(theta + theta2)) + k*(phi2 - theta2)]])    
    # Calculating angular rate derivatives
    #x3dx4d = np.array(np.linalg.inv(M)*(action.reshape(2,1) - C - friction*np.matrix([[x3],[x4]]))).squeeze() 
    x3dx4d = np.array(np.linalg.inv(M)*(action.reshape(2,1) - C - friction*np.matrix([[x3],[x4]]))).squeeze() 
    # Building the derivative matrix d[x1, x2, x3, x4]/dt = [x3, x4, x3dx4d]
    derivatives = np.concatenate((np.array([x3,x4]),x3dx4d)) 

    return derivatives

