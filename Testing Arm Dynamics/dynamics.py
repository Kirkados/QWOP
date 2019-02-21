
"""
This script provides the environment for the robotic arm simulation
"""
import numpy as np
import math
from scipy.integrate import odeint # numerical integrator (maybe call this in the dynamics function instead)

def equations_of_motion(state, t, args):
    Fsx, Fsy, Ms = args # unpacking inputs    
    x_p, x_p_dot, y_p, y_p_dot, theta_p, theta_p_dot, x_d, x_d_dot, y_d, y_d_dot, theta_d, theta_d_dot = state # unpacking state (theta1, theta2, theta1_dot, theta2_dot)
    
    
    

    M = np.matrix([[5*m*l**2/3 + m*l**2*math.cos(x2), m*l**2/3 + m*l**2*math.cos(x2)/2],[m*l**2/3 + m*l**2*math.cos(x2)/2, m*l**2/3]]) # mass matrix
    C = np.matrix([[-m*l**2*math.sin(x2)*x4*(2*x3+x4)],[m*l**2*x3**2*math.sin(x2)/2]])
    
    x3dx4d = np.array(np.linalg.inv(M)*(action - C - friction*np.matrix([[x3],[x4]]))) # calculating state derivative
    derivatives = np.concatenate((np.array((x3,x4)).reshape(2,1),x3dx4d)).squeeze() # packing results into derivative matrix

    return derivatives

def reward_function(self, action):
        
        return 0 # multiplying by the timestep to give the rewards on a per-second basis

class Environment():
    def __init__(self): # __init__ gets called immediately when the Environment is created.
        # Dumping information into self (which goes to all the subfunctions (methods) of Environment automatically)
        
        # Set initial conditions
        self.state = np.array([0., 0., 0., 0., np.pi/2, 0., 1 + 0.4*1, 0., 0., 0., np.pi/2, 0.]) # state: (x_p, x_p_dot, y_p, y_p_dot, theta_p, theta_p_dot, x_d, x_d_dot, y_d, y_d_dot, theta_d, theta_d_dot)
        
        self.initial_conditions = self.state # logging the initial conditions
        self.time = 0 # setting the initial time
        self.timestep = 0.1

        
    def one_timestep(self, action): # gets called when you use env.get_action(state). Already has access to the initialized self stuff from __init__

        parameters = [action]

        # integrating forward one time step. Returns initial condition on first row then next timestep on the next row
        next_states = odeint(equations_of_motion,self.state,[self.time, self.time + self.timestep], args = (parameters,), full_output = 0)
        #print(next_states)
        self.state = next_states[1,:] # remembering the current state
        self.time += self.timestep # updating the stored time
        
         
        # Taking sin and cos of angles so that the state appears to be from [-1 1]
        #trig_state = np.array([np.cos(self.state[0]), np.sin(self.state[0]), np.cos(self.state[1]), np.sin(self.state[1]), self.state[2], self.state[3]])
        #return np.append(trig_state, self.desired_ee_position), reward, terminate
        return self.state
    
    def initial_condition(self):
        # Returns the initial conditions if needed on the first timestep
        
        # Taking sin and cos of angles so that the state appears to be from [-1 1]
        #trig_state = np.array([np.cos(self.initial_conditions[0]), np.sin(self.initial_conditions[0]), np.cos(self.initial_conditions[1]), np.sin(self.initial_conditions[1]), self.initial_conditions[2], self.initial_conditions[3]])
        
        #return np.append(trig_state, self.desired_ee_position)
        return self.state

        