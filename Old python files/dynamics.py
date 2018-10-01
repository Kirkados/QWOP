
"""
This script provides the environment for the robotic arm simulation
"""
import numpy as np
import math
from scipy.integrate import odeint # numerical integrator (maybe call this in the dynamics function instead)

def equations_of_motion(state, t, parameters):
    l, m, action, friction = parameters # unpacking parameters
    x1, x2, x3, x4 = state # unpacking state (theta1, theta2, theta1_dot, theta2_dot)

    M = np.matrix([[5*m*l**2/3 + m*l**2*math.cos(x2), m*l**2/3 + m*l**2*math.cos(x2)/2],[m*l**2/3 + m*l**2*math.cos(x2)/2, m*l**2/3]]) # mass matrix
    C = np.matrix([[-m*l**2*math.sin(x2)*x4*(2*x3+x4)],[m*l**2*x3**2*math.sin(x2)/2]])
    
    x3dx4d = np.array(np.linalg.inv(M)*(action - C - friction*np.matrix([[x3],[x4]]))) # calculating state derivative
    derivatives = np.concatenate((np.array((x3,x4)).reshape(2,1),x3dx4d)).squeeze() # packing results into derivative matrix

    return derivatives

def reward_function(self, action):
        reward = 0 # initializing
        
        # Giving rewards for being near the target location
        current_ee_position = np.array([self.length*np.cos(self.state[0]) + self.length*np.cos(self.state[0] + self.state[1]), self.length*np.sin(self.state[0]) + self.length*np.sin(self.state[0] + self.state[1])])
        
        if self.shape_rewards: # if we want to shape the success rewards
            reward += np.exp(-np.linalg.norm(self.desired_ee_position - current_ee_position)) * self.target_reward
                
        else: # binary success rewards                            
            if np.linalg.norm(self.desired_ee_position - current_ee_position) < self.reward_circle_radius:
                reward += self.target_reward
        
        # Giving rewards (or penalties) for using joint torque
        reward += np.abs(action[0])*self.torque_reward + np.abs(action[1])*self.torque_reward
        
        # Giving penalties for having high angular rates
        reward += np.abs(self.state[2])*self.ang_rate_reward + np.abs(self.state[2] + self.state[3]) * self.ang_rate_reward
            
        return reward*self.timestep # multiplying by the timestep to give the rewards on a per-second basis

def ee_position_error(self):
    current_ee_position = np.array([self.length*np.cos(self.state[0]) + self.length*np.cos(self.state[0] + self.state[1]), self.length*np.sin(self.state[0]) + self.length*np.sin(self.state[0] + self.state[1])])
    #print(current_ee_position - self.desired_ee_position)
    return current_ee_position - self.desired_ee_position

class Environment():
    def __init__(self, timestep, mass, length, target_reward, torque_reward, ang_rate_reward, reward_circle_radius, reward_shaping, max_ang_rate, randomize, friction): # __init__ gets called immediately when the Environment is created.
        # Dumping information into self (which goes to all the subfunctions (methods) of Environment automatically)
        
        # I'll keep the initial conditions in the self so that I don't need to keep passing them back in here from main.
        if randomize: # start with random initial conditions
            self.state = np.array([np.random.rand(1)[0]*2*np.pi, np.random.rand(1)[0]*2*np.pi, 0, 0]) # randomizing the initial state of the manipulator
        else: # start with consistent initial conditions
            self.state = np.array([np.pi/2, -np.pi/2, 0., 0.])
        self.initial_conditions = self.state # logging the initial conditions
        self.time = 0 # setting the initial time
        self.timestep = timestep # holding the desired integration timestep
        self.length = length
        self.mass = mass
        self.target_reward = target_reward
        self.torque_reward = torque_reward
        self.ang_rate_reward = ang_rate_reward
        self.reward_circle_radius = reward_circle_radius # [m] sparse circle where rewards are given
        self.shape_rewards = reward_shaping # whether or not to shape the rewards
        self.max_ang_rate = max_ang_rate # maximum angular rate allowed before termination
        self.friction = friction # friction present in the joints
        
        if randomize: # random target location
            desired_r = np.random.rand(1)[0]*2*length # random point on reachable circle
            desired_angle = np.random.rand(1)[0]*2*np.pi # random angle on circle
        else: # consistent target location
            desired_r = 1
            desired_angle = np.pi
        self.desired_ee_position = np.array([desired_r*np.cos(desired_angle), desired_r*np.sin(desired_angle)])
        
    def one_timestep(self, action): # gets called when you use env.get_action(state). Already has access to the initialized self stuff from __init__
        
        parameters = [self.length, self.mass, action, self.friction]

        # integrating forward one time step. Returns initial condition on first row then next timestep on the next row
        next_states = odeint(equations_of_motion,self.state,[self.time, self.time + self.timestep], args = (parameters,), full_output = 0)
        #print(next_states)
        self.state = next_states[1,:] # remembering the current state
        self.time += self.timestep # updating the stored time
        
        reward = reward_function(self, action)
        
        terminate = False
        # if things are getting out of control
        if (np.abs(self.state[2]) > self.max_ang_rate) | (np.abs(self.state[2] + self.state[3]) > self.max_ang_rate):
            print('Too fast! Terminating early')
            terminate = True
        
        # Taking sin and cos of angles so that the state appears to be from [-1 1]
        #trig_state = np.array([np.cos(self.state[0]), np.sin(self.state[0]), np.cos(self.state[1]), np.sin(self.state[1]), self.state[2], self.state[3]])
        #return np.append(trig_state, self.desired_ee_position), reward, terminate
        return np.append(self.state, ee_position_error(self)), reward, terminate
    
    def initial_condition(self):
        # Returns the initial conditions if needed on the first timestep
        
        # Taking sin and cos of angles so that the state appears to be from [-1 1]
        #trig_state = np.array([np.cos(self.initial_conditions[0]), np.sin(self.initial_conditions[0]), np.cos(self.initial_conditions[1]), np.sin(self.initial_conditions[1]), self.initial_conditions[2], self.initial_conditions[3]])
        
        #return np.append(trig_state, self.desired_ee_position)
        return np.append(self.state, ee_position_error(self))
    
    def target_location(self):
        return self.desired_ee_position
        