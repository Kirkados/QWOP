
"""
This script generates the environment for reinforcement learning agents.

A simplified QWOP agent is modelled. It has one torso and two legs.

The agent is encouraged to travel down the track. Its reward is proportional
to its forward velocity.

All dynamic environments I create will have a standardized architecture. The 
reason for this is I have one learning algorithm and many environments. All 
environments are responsible for:
    - dynamics propagation (via the step method)
    - initial conditions   (via the reset method)
    - reporting environment properties (defined in __init__)
    - seeding the dynamics (via the seed method)
          
Outputs:
    Reward must be of shape ()
    State must be of shape (state_size,)
    Done must be a bool

Inputs:
    Action input is of shape (action_size,)   


@author: Kirk Hovell (khovell@gmail.com)
"""
import numpy as np
from scipy.integrate import odeint # Numerical integrator

class Environment:
    
    def __init__(self): 
        ##################################
        ##### Environment Properties #####
        ##################################
        self.state_size = 18
        self.action_size = 2
        self.timestep = 0.1 # [s]
        self.target_reward = 1.
        self.num_frames = 100 # total animation is cut into this many frames
        self.randomize = False # whether or not to randomize the state & target location
        
        # To be removed
        self.lower_action_bound = np.array([-0.1, -0.1]) # [Nm]
        self.upper_action_bound = np.array([ 0.1,  0.1]) # [Nm]
        self.lower_state_bound =  np.array([1., 1., 1.])
        self.upper_state_bound =  np.array([1., 1., 1.])
        
        # Physical Properties
        self.m = 10 # [kg]
        self.m1 = 5 # [kg]
        self.m2 = 5 # [kg]
        self.eta = 0.5 
        self.eta1 = 0.5
        self.eta2 = 0.5
        self.gamma1 = 0.5
        self.gamma2 = 0.5
        self.I = 10
        self.I1 = 7
        self.I2 = 7
        self.g = 9.81
        self.body_length = 1. # [m]
        self.leg1_length = 1. # [m]
        self.leg2_length = 1. # [m]
        self.phi1 = np.pi/6 # [rad]
        self.phi2 = -np.pi/6 # [rad]

        
    ######################################
    ##### Resettings the Environment #####
    ######################################    
    def reset(self):
        # This method resets the state and returns it
        
        # If we are randomizing the initial consitions and state
        if self.randomize: 
            # Randomizing initial conditions 
            pass # to be completed later
            
        else:
            # Consistent initial conditions
            # Define the initial angles and torso height and the remainder is calculated
            initial_body_angle = 0.
            initial_leg1_angle = np.pi/6
            initial_leg2_angle = -np.pi/6
            initial_torso_height = 2. # [m] above ground
            
            # Calculating leg initial positions
            initial_x1 = self.body_length * self.eta * np.sin(initial_body_angle) + self.leg1_length * self.eta1 * np.sin(initial_body_angle + initial_leg1_angle)
            initial_y1 = initial_torso_height - self.body_length * self.eta * np.cos(initial_body_angle) - self.leg1_length * self.eta1 * np.cos(initial_body_angle + initial_leg1_angle)
            initial_x2 = self.body_length * self.eta * np.sin(initial_body_angle) + self.leg2_length * self.eta2 * np.sin(initial_body_angle + initial_leg2_angle)
            initial_y2 = initial_torso_height - self.body_length * self.eta * np.cos(initial_body_angle) - self.leg2_length * self.eta2 * np.cos(initial_body_angle + initial_leg2_angle)
            
            # Assembling into the state
            # Note: state = [x, y, theta, x1, y1, theta1, x2, y2, theta2, xdot, ydot, thetadot, x1dot, y1dot, theta1dot, x2dot, y2dot, theta2dot]
            self.state = np.array([0., initial_torso_height, initial_body_angle, initial_x1, initial_y1, initial_leg1_angle, initial_x2, initial_y2, initial_leg2_angle, 0., 0., 0., 0., 0., 0., 0., 0., 0.,])

        # Resetting the time
        self.time = 0.         
        
        # Return the state
        return self.state
        
    #####################################
    ##### Step the Dynamics forward #####
    #####################################
    def step(self, action):
        
        # Initializing
        done = False  
        
        # Calculating desired leg angles 
        action = action.reshape(2,1)         
        # Incrementing the desired leg angles
        self.phi1 += action[0]
        self.phi2 += action[1]          
        
        # Choosing friction and normal force
        fF1 = 0.
        fF2 = 0.
        fN1 = 0.
        fN2 = 0.
        K = 100.
        
        # Packing up the parameters the equations of motion need
        parameters = [self.m, self.m1, self.m2, self.eta, self.eta1, self.eta2, self.gamma1, self.gamma2, self.I, self.I1, self.I2, self.g, fF1, fF2, self.phi1, self.phi2, fN1, fN2, K]

        # Integrating forward one time step. 
        # Returns initial condition on first row then next timestep on the next row
        ##############################
        ##### PROPAGATE DYNAMICS #####
        ##############################
        next_states = odeint(equations_of_motion,self.state,[self.time, self.time + self.timestep], args = (parameters,), full_output = 0)
        
        reward = self.reward_function(action) 
        
        self.state = next_states[1,:] # remembering the current state
        self.time += self.timestep # updating the stored time
        

        # Return the (state, reward, done)
        return self.state, reward, done

    
    def seed(self, seed = None):
        # This method seeds the environment, for reproducability
        np.random.seed(seed)


    def reward_function(self, action):
        # Returns the reward for this timestep as a function of the state and action
        
        # The agent is (currently) rewarded for forward velocity.
        reward = self.state[9]        
        return reward

def equations_of_motion(state, t, parameters):
    # From the state, it returns the first derivative of the state
    
    # Unpacking the state
    x, y, theta, x1, y1, theta1, x2, y2, theta2, xdot, ydot, thetadot, x1dot, y1dot, theta1dot, x2dot, y2dot, theta2dot = state
    
    # Unpacking parameters
    m, m1, m2, eta, eta1, eta2, gamma1, gamma2, I, I1, I2, g, fF1, fF2, phi1, phi2, fN1, fN2, K = parameters 
    
    first_derivatives = np.array([xdot, ydot, thetadot, x1dot, y1dot, theta1dot, x2dot, y2dot, theta2dot])

    # Mass matrix
    M = np.matrix([
                   [                    m,                   0.,                                                0.,                               m1,                               0.,                            0.,                               m2,                               0.,                            0.],
                   [                   0.,                   m ,                                                0.,                               0.,                               m1,                            0.,                               0.,                               m2,                            0.],
                   [ -m*eta*np.cos(theta), -m*eta*np.sin(theta),                                                 I,                               0.,                               0.,                            0.,                               0.,                               0.,                            0.],
                   [                  -1.,                   0.,          eta*np.sin(theta) + gamma1*np.cos(theta),                               1.,                               0., gamma1*np.sin(theta + theta1),                               0.,                               0.,                            0.],
                   [                   0.,                  -1., eta*np.cos(theta) + gamma1*np.cos(theta + theta1),                               0.,                               1., gamma1*np.cos(theta + theta1),                               0.,                               0.,                            0.],
                   [                   0.,                   0.,                                                0., m1*gamma1*np.cos(theta + theta1), m1*gamma1*np.sin(theta + theta1),                            I1,                               0.,                               0.,                            0.],
                   [                  -1.,                   0., eta*np.sin(theta) + gamma2*np.sin(theta + theta2),                               0.,                               0.,                            0.,                               1.,                               0., gamma2*np.sin(theta + theta2)],
                   [                   0.,                  -1., eta*np.cos(theta) + gamma2*np.cos(theta + theta2),                               0.,                               0.,                            0.,                               0.,                               1., gamma2*np.cos(theta + theta2)],
                   [                   0.,                   0.,                                                0.,                               0.,                               0.,                            0., m2*gamma2*np.cos(theta + theta2), m2*gamma2*np.sin(theta + theta2),                            I2]])
    
    
    # C matrix
    C = np.matrix([[fF1 + fF2],
                   [fN1 + fN2 - (m + m1 + m2)*g],
                   [-K*(phi1 - theta1 + phi2 - theta2) + m*g*np.sin(theta)],
                   [-thetadot**2*eta*np.cos(theta) - (thetadot + theta1dot)**2*gamma1*np.cos(theta + theta1)],
                   [ thetadot**2*eta*np.sin(theta) + (thetadot + theta1dot)**2*gamma1*np.sin(theta + theta1)],
                   [-g*gamma1*np.sin(theta+theta1) + fN1*(eta1*np.sin(theta+theta1) + gamma1*np.sin(theta+theta1)) + fF1*(eta1*np.cos(theta+theta1) + gamma1*np.cos(theta + theta1)) + K*(phi1 - theta1)],
                   [-thetadot**2*eta*np.cos(theta) - (thetadot + theta2dot)**2*gamma2*np.cos(theta+theta2)],
                   [thetadot**2*eta*np.sin(theta) + (thetadot + theta2dot)**2*gamma2*np.sin(theta + theta2)],
                   [-g*gamma2*np.sin(theta + theta2) + fN2*(eta2*np.sin(theta + theta2) + gamma2*np.sin(theta + theta2)) + fF2*(eta2*np.cos(theta + theta2) + gamma2*np.cos(theta + theta2)) + K*(phi2 - theta2)]])    
    
    # Calculating angular rate derivatives
    #x3dx4d = np.array(np.linalg.inv(M)*(action.reshape(2,1) - C - friction*np.matrix([[x3],[x4]]))).squeeze() 
    second_derivatives = np.array(np.linalg.inv(M)*(C)).squeeze() 
    
    # Building the derivative matrix d[state]/dt = [first_derivatives, second_derivatives]
    #print(np.concatenate((first_derivatives, second_derivatives)))
    #raise SystemExit
    derivatives = np.concatenate((first_derivatives, second_derivatives)) 

    return derivatives

# Mass matrix
#    M = np.matrix([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                    m,                   0.,                                                0.,                               m1,                               0.,                            0.,                               m2,                               0.,                            0.],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                   0.,                   m ,                                                0.,                               0.,                               m1,                            0.,                               0.,                               m2,                            0.],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 0., -m*eta*np.cos(theta), -m*eta*np.sin(theta),                                                 I,                               0.,                               0.,                            0.,                               0.,                               0.,                            0.],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                  -1.,                   0.,          eta*np.sin(theta) + gamma1*np.cos(theta),                               1.,                               0., gamma1*np.sin(theta + theta1),                               0.,                               0.,                            0.],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                   0.,                  -1., eta*np.cos(theta) + gamma1*np.cos(theta + theta1),                               0.,                               1., gamma1*np.cos(theta + theta1),                               0.,                               0.,                            0.],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                   0.,                   0.,                                                0., m1*gamma1*np.cos(theta + theta1), m1*gamma1*np.sin(theta + theta1),                            I1,                               0.,                               0.,                            0.],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                  -1.,                   0., eta*np.sin(theta) + gamma2*np.sin(theta + theta2),                               0.,                               0.,                            0.,                               1.,                               0., gamma2*np.sin(theta + theta2)],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                   0.,                  -1., eta*np.cos(theta) + gamma2*np.cos(theta + theta2),                               0.,                               0.,                            0.,                               0.,                               1., gamma2*np.cos(theta + theta2)],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 0.,                   0.,                   0.,                                                0.,                               0.,                               0.,                            0., m2*gamma2*np.cos(theta + theta2), m2*gamma2*np.sin(theta + theta2),                            I2]])
 
# C matrix
#    C = np.matrix([[xdot],
#                   [ydot],
#                   [thetadot],
#                   [x1dot],
#                   [y1dot],
#                   [theta1dot],
#                   [x2dot],
#                   [y2dot],
#                   [theta2dot],
#                   [fF1 + fF2],
#                   [fN1 + fN2 - (m + m1 + m2)*g],
#                   [-K*(phi1 - theta1 + phi2 - theta2) + m*g*np.sin(theta)],
#                   [-thetadot**2*eta*np.cos(theta) - (thetadot + theta1dot)**2*gamma1*np.cos(theta + theta1)],
#                   [ thetadot**2*eta*np.sin(theta) + (thetadot + theta1dot)**2*gamma1*np.sin(theta + theta1)],
#                   [-g*gamma1*np.sin(theta+theta1) + fN1*(eta1*np.sin(theta+theta1) + gamma1*np.sin(theta+theta1)) + fF1*(eta1*np.cos(theta+theta1) + gamma1*np.cos(theta + theta1)) + K*(phi1 - theta1)],
#                   [-thetadot**2*eta*np.cos(theta) - (thetadot + theta2dot)**2*gamma2*np.cos(theta+theta2)],
#                   [thetadot**2*eta*np.sin(theta) + (thetadot + theta2dot)**2*gamma2*np.sin(theta + theta2)],
#                   [-g*gamma2*np.sin(theta + theta2) + fN2*(eta2*np.sin(theta + theta2) + gamma2*np.sin(theta + theta2)) + fF2*(eta2*np.cos(theta + theta2) + gamma2*np.cos(theta + theta2)) + K*(phi2 - theta2)]])    