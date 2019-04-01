
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
import multiprocessing
import signal

from scipy.integrate import odeint # Numerical integrator

class Environment:
    # For reference
    # state = x, y, theta, x1, y1, theta1, x2, y2, theta2, xf1, yf1, xf2, yf2, xdot, ydot, thetadot, x1dot, y1dot, theta1dot, x2dot, y2dot, theta2dot, xf1dot, yf1dot, xf2dot, yf2dot
    
    def __init__(self): 
        ##################################
        ##### Environment Properties #####
        ##################################
        self.TOTAL_STATE_SIZE        = 26 # total number of states
        self.IRRELEVANT_STATES       = [0,3,4,6,7,9,10,11,12,16,17,19,20,22,23,24,25] # states that are irrelevant to the policy network in its decision making
        self.STATE_SIZE              = self.TOTAL_STATE_SIZE - len(self.IRRELEVANT_STATES) # total number of relevant states
        self.ACTION_SIZE             = 9
        self.TIMESTEP                = 0.05 # [s]        
        self.MAX_NUMBER_OF_TIMESTEPS = 1200 # per episode
        self.NUM_FRAMES              = 100 # total animation is cut into this many frames
        self.RANDOMIZE               = False # whether or not to randomize the state & target location
        self.UPPER_STATE_BOUND       = np.array([np.inf, 2., 2*np.pi, np.inf, np.inf, 2*np.pi, np.inf, np.inf, 2*np.pi, np.inf, np.inf, np.inf, np.inf, 1., 1., 1., np.inf, np.inf, 1., np.inf, np.inf, 1., np.inf, np.inf, np.inf, np.inf])
        self.NORMALIZE_STATE         = True # Normalize state on each timestep to avoid vanishing gradients
        self.REWARD_SCALING          = 1000.0 # Amount to scale down the reward signal
        self.MIN_Q                   = -5.0
        self.MAX_Q                   = 15.0
        self.DONE_ON_FALL            = True # whether or not falling down ends the episode
        
        # Rendering parameters
        self.HEIGHT = 500
        self.WIDTH  = 800
        self.HUMAN_SCALE = 150 #pixel/m
        
        self.x_0 = np.rint(self.WIDTH/2) # 0 in x on the screen
        self.y_0 = np.rint(self.HEIGHT*9/10) # 0 in y on the screen
        
        # How much the leg desired angle changes per frame when a button is pressed
        self.HIP_INCREMENT        = 2.*np.pi/180. # [rad/s]
        self.HIP_SPRING_STIFFNESS = 1000 # [Nm/rad]       
        self.HIP_DAMPING_STIFFNESS = 100 # [Nm/rad]
        self.PHI1_INITIAL                 = 30*np.pi/180
        self.PHI2_INITIAL                 = -30*np.pi/180
        
        #friction properties
        self.FLOOR_MU               = 0.3
        self.FLOOR_SPRING_STIFFNESS = 100000 #[N/m]
        self.FLOOR_DAMPING_COEFFICIENT = 10000 # [Ns/m]
        self.FLOOR_FRICTION_STIFFNESS = 100000 #[N/m]
        self.ATANDEL = 0.001
        self.ATANGAIN = 12.7
        
        #self.body.segment(0) = 5
        
        self.g = 9.81         # [m/s^2]

        # Physical Properties
        self.SEGMENT_MASS = list()# [kg]
        self.SEGMENT_MOMINERT = list()# [kg m^2]
        self.SEGMENT_LENGTH = list()# [m]
        self.SEGMENT_ETA_LENGTH = list()# [m]
        self.SEGMENT_GAMMA_LENGTH = list()# [m]
        self.SEGMENT_PHI_NAUGTH = list()# [rad]
        self.SEGMENT_PHI_MAX = list()# [rad]
        self.SEGMENT_PHI_MIN = list()# [rad]
        
        
        #player stats
        p_mass = 75#kg
        p_height = 1.80#m

        # body 0, main body
        m = 0.497*p_mass
        L = 0.289*p_height
        eta = 0.5*L
        gamma = L-eta
        phi0 = 0 
        phi_min = -1000000
        phi_max =  1000000
        k = 0.5*L
        I = m*k**2
        
         
        self.SEGMENT_MASS.append(m)
        self.SEGMENT_MOMINERT.append(I)
        self.SEGMENT_LENGTH.append(L)
        self.SEGMENT_ETA_LENGTH.append(eta)
        self.SEGMENT_GAMMA_LENGTH.append(gamma)
        self.SEGMENT_PHI_NAUGTH.append(phi0)
        self.SEGMENT_PHI_MAX.append(phi_max)
        self.SEGMENT_PHI_MIN.append(phi_min)
        
        # body 1, right leg
        m = 0.0465*p_mass+0.100*p_mass
        L = 0.242*p_height+0.245*p_height
        eta = (0.433*0.245*p_height*0.100*p_mass+(0.245*p_height+0.433*0.242*p_height)*0.0465*p_mass)/m
        gamma = L-eta
        phi0 = np.pi/6 
        phi_min = -np.pi/4
        phi_max =  np.pi/2
        k = 0.5*L
        I = 0.0465*p_mass*(0.302*0.245*p_height**2)+0.0465*p_mass*(0.323*0.242*p_height**2)
       
        self.SEGMENT_MASS.append(m)
        self.SEGMENT_MOMINERT.append(I)
        self.SEGMENT_LENGTH.append(L)
        self.SEGMENT_ETA_LENGTH.append(eta)
        self.SEGMENT_GAMMA_LENGTH.append(gamma)
        self.SEGMENT_PHI_NAUGTH.append(phi0)
        self.SEGMENT_PHI_MAX.append(phi_max)
        self.SEGMENT_PHI_MIN.append(phi_min)
        
        # body 2, left leg
        m = 0.0465*p_mass+0.100*p_mass
        L = 0.242*p_height+0.245*p_height
        eta = (0.433*0.245*p_height*0.100*p_mass+(0.245*p_height+0.433*0.242*p_height)*0.0465*p_mass)/m
        gamma = L-eta
        phi0 = -np.pi/6 
        phi_min = -np.pi/4
        phi_max =  np.pi/2
        k = 0.5*L
        I = 0.0465*p_mass*(0.302*0.245*p_height**2)+0.0465*p_mass*(0.323*0.242*p_height**2)
        
        self.SEGMENT_MASS.append(m)
        self.SEGMENT_MOMINERT.append(I)
        self.SEGMENT_LENGTH.append(L)
        self.SEGMENT_ETA_LENGTH.append(eta)
        self.SEGMENT_GAMMA_LENGTH.append(gamma)
        self.SEGMENT_PHI_NAUGTH.append(phi0)
        self.SEGMENT_PHI_MAX.append(phi_max)
        self.SEGMENT_PHI_MIN.append(phi_min)
        
#        self.m = 10.          
#        self.m1 = 5.          # [kg]
#        self.m2 = 5.          # [kg]
#        self.eta = 0.5 
#        self.eta1 = 0.5
#        self.eta2 = 0.5
#        self.gamma1 = 0.5
#        self.gamma2 = 0.5
#        self.I = 10.          
#        self.I1 = 7.          # [kg m^2]
#        self.I2 = 7.          # [kg m^2]
#        self.body_length = 1. 
#        self.leg1_length = 1. # [m]
#        self.leg2_length = 1. # [m]
#        self.phi1 = np.pi/6   # [rad]
#        self.phi2 = -np.pi/6  # [rad]

        
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
        if self.RANDOMIZE: 
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
            initial_x1 = self.SEGMENT_ETA_LENGTH[0] * np.sin(initial_body_angle) + self.SEGMENT_GAMMA_LENGTH[1] * np.sin(initial_body_angle + initial_leg1_angle)
            initial_y1 = initial_torso_height - self.SEGMENT_ETA_LENGTH[0] * np.cos(initial_body_angle) - self.SEGMENT_GAMMA_LENGTH[1] * np.cos(initial_body_angle + initial_leg1_angle)
            initial_x2 = self.SEGMENT_ETA_LENGTH[0] * np.sin(initial_body_angle) + self.SEGMENT_GAMMA_LENGTH[2] * np.sin(initial_body_angle + initial_leg2_angle)
            initial_y2 = initial_torso_height - self.SEGMENT_ETA_LENGTH[0] * np.cos(initial_body_angle) - self.SEGMENT_GAMMA_LENGTH[2] * np.cos(initial_body_angle + initial_leg2_angle)
            
            initial_xf1 = self.SEGMENT_ETA_LENGTH[0] * np.sin(initial_body_angle) + self.SEGMENT_LENGTH[1] * np.sin(initial_body_angle + initial_leg1_angle)
            initial_yf1 = initial_torso_height - self.SEGMENT_ETA_LENGTH[0] * np.cos(initial_body_angle) - self.SEGMENT_LENGTH[1] * np.cos(initial_body_angle + initial_leg1_angle)
            initial_xf2 = self.SEGMENT_ETA_LENGTH[0] * np.sin(initial_body_angle) + self.SEGMENT_LENGTH[1] * np.sin(initial_body_angle + initial_leg1_angle)
            initial_yf2 = initial_torso_height - self.SEGMENT_ETA_LENGTH[0] * np.cos(initial_body_angle) - self.SEGMENT_LENGTH[1] * np.cos(initial_body_angle + initial_leg1_angle)
            
            self.phi1 = self.PHI1_INITIAL 
            self.phi2 = self.PHI2_INITIAL 
            # Assembling into the state
            # Note: state = [x, y, theta, x1, y1, theta1, x2, y2, theta2, xdot, ydot, thetadot, x1dot, y1dot, theta1dot, x2dot, y2dot, theta2dot]
            self.state = np.array([0., initial_torso_height, initial_body_angle, initial_x1, initial_y1, initial_leg1_angle, initial_x2, initial_y2, initial_leg2_angle, initial_xf1,initial_yf1,initial_xf2,initial_yf2,0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.,])

        # Resetting the time
        self.time = 0.0  
        
        # Return the state
        return self.state
        
    
    def parse_action(self,action):
        #0: No buttons pressed; 1: Q only; 2: QO; 3: QP; 4: W only; 5: WO; 6: WP; 7: O only; 8: P only
        
        pressed_q = False
        pressed_w = False
        pressed_o = False
        pressed_p = False
        
        if action == 1: 
            pressed_q = True
        elif action == 2: 
            pressed_q = True
            pressed_o = True
        elif action == 3: 
            pressed_q = True
            pressed_p = True
        elif action == 4: 
            pressed_w = True
        elif action == 5: 
            pressed_w = True
            pressed_o = True
        elif action == 6: 
            pressed_w = True
            pressed_p = True
        elif action == 7:
            pressed_o = True
        elif action == 8: 
            pressed_p = True
        
        return pressed_q, pressed_w, pressed_o, pressed_p
       
    # Returns the point coordinates for each member 
    def returnPointCoords(self, x, y, theta_cum, gamma, eta, x_c, x_0, y_0, hum_scale):
        #gamma is "above", eta is "below" CG
        
        pointCoords = np.array([[x-gamma*np.sin(theta_cum),y+gamma*np.cos(theta_cum)],[x,y],[x+eta*np.sin(theta_cum),y-eta*np.cos(theta_cum)]])      
        
        pointCoords[:,0]=(pointCoords[:,0]-x_c)*hum_scale+x_0
        pointCoords[:,1]=y_0-(pointCoords[:,1])*hum_scale
        
        return pointCoords
    
    def is_done(self):
        # Determines whether this episode is done
        
        # Define the body
        segment_count = 3
        segment_points = np.zeros((segment_count,3,2))
        
        # Unpacking the state
        x, y, theta, x1, y1, theta1, x2, y2, theta2, *_ = self.state

        #Get point coordinates for each segment
        segment_points[0,:,:] = self.returnPointCoords(x,y,theta,self.SEGMENT_GAMMA_LENGTH[0],self.SEGMENT_ETA_LENGTH[0],x,self.x_0,self.y_0,self.HUMAN_SCALE)
        segment_points[1,:,:] = self.returnPointCoords(x1,y1,theta+theta1,self.SEGMENT_GAMMA_LENGTH[1],self.SEGMENT_ETA_LENGTH[1],x,self.x_0,self.y_0,self.HUMAN_SCALE)
        segment_points[2,:,:] = self.returnPointCoords(x2,y2,theta+theta2,self.SEGMENT_GAMMA_LENGTH[2],self.SEGMENT_ETA_LENGTH[2],x,self.x_0,self.y_0,self.HUMAN_SCALE)
        
        # Check if any node is out-of-bounds. If so, this episode is done
        if np.any(segment_points[:,0,1] > self.HEIGHT) and self.DONE_ON_FALL:
            done = True
            #print("Episode done at time %.2f seconds" %self.time)
        else:
            done = False
        
        return done
    
    
    #####################################
    ##### Step the Dynamics forward #####
    #####################################
    def step(self, action):
        
        # Parsing action number into button presses
        #0: No buttons pressed; 1: Q only; 2: QO; 3: QP; 4: W only; 5: WO; 6: WP; 7: O only; 8: P only
        pressed_q, pressed_w, pressed_o, pressed_p = self.parse_action(action)

        # Incrementing the desired leg angles
        # Q and W control leg 1 through phi1
        # O and P control leg 2 through phi2
        if pressed_q:
            self.phi1 += self.HIP_INCREMENT
            self.phi1 = np.minimum(self.phi1,self.SEGMENT_PHI_MAX[1])
        if pressed_w:
            self.phi1 -= self.HIP_INCREMENT
            self.phi1 = np.maximum(self.phi1,self.SEGMENT_PHI_MIN[1])
        if pressed_o:
            self.phi2 += self.HIP_INCREMENT
            self.phi1 = np.minimum(self.phi2,self.SEGMENT_PHI_MAX[2])
        if pressed_p:
            self.phi2 -= self.HIP_INCREMENT 
            self.phi1 = np.maximum(self.phi2,self.SEGMENT_PHI_MIN[2])  
        
        # Choosing friction and normal force
        fF1 = 0.
        fF2 = 0.
        fN1 = 0.
        fN2 = 0.
        
        # Packing up the parameters the equations of motion need
        parameters = np.array([self.SEGMENT_MASS[0],self.SEGMENT_MASS[1],self.SEGMENT_MASS[2], self.SEGMENT_LENGTH[0],self.SEGMENT_LENGTH[1],self.SEGMENT_LENGTH[2],self.SEGMENT_ETA_LENGTH[0],self.SEGMENT_ETA_LENGTH[1],self.SEGMENT_ETA_LENGTH[2], self.SEGMENT_GAMMA_LENGTH[0],self.SEGMENT_GAMMA_LENGTH[1],self.SEGMENT_GAMMA_LENGTH[2], self.SEGMENT_MOMINERT[0], self.SEGMENT_MOMINERT[1], self.SEGMENT_MOMINERT[2], self.g, fF1, fF2, self.phi1, self.phi2, fN1, fN2, self.HIP_SPRING_STIFFNESS, self.HIP_DAMPING_STIFFNESS, self.FLOOR_SPRING_STIFFNESS,self.FLOOR_FRICTION_STIFFNESS, self.FLOOR_MU, self.ATANDEL,self.ATANGAIN, self.FLOOR_DAMPING_COEFFICIENT], dtype = 'float64')

        # Integrating forward one time step. 
        # Returns initial condition on first row then next timestep on the next row
        ##############################
        ##### PROPAGATE DYNAMICS #####
        ##############################
        next_states = odeint(equations_of_motion, self.state, [self.time, self.time + self.TIMESTEP], args = (parameters,), full_output = 0)
        
        # Get this timestep's reward
        reward = self.reward_function(action) 
        
        # Check if this episode is done
        done = self.is_done()
        
        self.state = next_states[1,:] # remembering the current state
        self.time += self.TIMESTEP # updating the stored time
        

        # Return the (state, reward, done)
        return self.state, reward, done


    def reward_function(self, action):
        # Returns the reward for this timestep as a function of the state and action
        
        # The agent is (currently) rewarded for forward velocity.
        reward = self.state[13]        
        return reward
    
    
    #########################################################################
    ##### Generating communication links between environment and agents #####
    #########################################################################
    def generate_queue(self):
        # Generate the queues responsible for communicating with the agent
        self.agent_to_env = multiprocessing.Queue(maxsize = 1)  
        self.env_to_agent = multiprocessing.Queue(maxsize = 1)  
        return self.agent_to_env, self.env_to_agent
        
    

    ###################################
    ##### Running the environment #####
    ###################################    
    def run(self):

        """
        This method is called when the environment process is launched by main.py.
        It is responsible for continually listening for an input action from the
        agent through a Queue. If an action is received, it is to step the environment
        and return the results.
        """
        # Instructing this process to treat Ctrl+C events (called SIGINT) by going SIG_IGN (ignore).
        # This permits the process to continue upon a Ctrl+C event to allow for graceful quitting.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
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
        





def equations_of_motion(state, t, parameters):
    # From the state, it returns the first derivative of the state
    
    # Unpacking the state
    x, y, theta, x1, y1, theta1, x2, y2, theta2, xf1,yf1,xf2,yf2, xdot, ydot, thetadot, x1dot, y1dot, theta1dot, x2dot, y2dot, theta2dot,xf1dot,yf1dot,xf2dot,yf2dot = state
    
    # Unpacking parameters
    m, m1, m2, l, l1, l2, eta, eta1, eta2, gamma, gamma1, gamma2, I, I1, I2, g, fF1, fF2, phi1, phi2, fN1, fN2, HIP_SPRING_STIFFNESS,HIP_DAMPING_STIFFNESS, FLOOR_SPRING_STIFFNESS, FLOOR_FRICTION_STIFFNESS, FLOOR_MU, ATANDEL, ATANGAIN, FLOOR_DAMPING_COEFFICIENT  = parameters 
    
    first_derivatives = np.array([xdot, ydot, thetadot, x1dot, y1dot, theta1dot, x2dot, y2dot, theta2dot, xf1dot, yf1dot, xf2dot, yf2dot])

    # Mass matrix
    M = np.matrix([#                    x                    y                                                 theta                              x1                                y1                             theta1                            x2                                y2                             theta2                           xf1                            yf1                              xf2                            yf2
                   [                    m,                   0.,                                                0.,                               m1,                               0.,                            0.,                               m2,                               0.,                            0.,                              0.,                            0.,                               0.,                            0.],
                   [                   0.,                   m ,                                                0.,                               0.,                               m1,                            0.,                               0.,                               m2,                            0.,                              0.,                            0.,                               0.,                            0.],
                   [ -m*eta*np.cos(theta), -m*eta*np.sin(theta),                                                 I,                               0.,                               0.,                            0.,                               0.,                               0.,                            0.,                              0.,                            0.,                               0.,                            0.],
                   [                   1.,                   0., eta*np.cos(theta) + gamma1*np.cos(theta + theta1),                              -1.,                               0., gamma1*np.cos(theta + theta1),                               0.,                               0.,                            0.,                              0.,                            0.,                               0.,                            0.],
                   [                   0.,                   1., eta*np.sin(theta) + gamma1*np.sin(theta + theta1),                               0.,                              -1., gamma1*np.sin(theta + theta1),                               0.,                               0.,                            0.,                              0.,                            0.,                               0.,                            0.],
                   [                   0.,                   0.,                                                0., m1*gamma1*np.cos(theta + theta1), m1*gamma1*np.sin(theta + theta1),                            I1,                               0.,                               0.,                            0.,                              0.,                            0.,                               0.,                            0.],
                   [                   1.,                   0., eta*np.cos(theta) + gamma2*np.cos(theta + theta2),                               0.,                               0.,                            0.,                              -1.,                               0., gamma2*np.cos(theta + theta2),                              0.,                            0.,                               0.,                            0.],
                   [                   0.,                   1., eta*np.sin(theta) + gamma2*np.sin(theta + theta2),                               0.,                               0.,                            0.,                               0.,                              -1., gamma2*np.sin(theta + theta2),                              0.,                            0.,                               0.,                            0.],
                   [                   0.,                   0.,                                                0.,                               0.,                               0.,                            0., m2*gamma2*np.cos(theta + theta2), m2*gamma2*np.sin(theta + theta2),                            I2,                              0.,                            0.,                               0.,                            0.],
                   [                   0.,                   0.,                      -eta1*np.cos(theta + theta1),                               -1,                               0.,  -eta1*np.cos(theta + theta1),                               0.,                               0.,                            0.,                              1 ,                            0.,                               0.,                            0.],
                   [                   0.,                   0.,                      -eta1*np.sin(theta + theta1),                               0.,                               -1,  -eta1*np.sin(theta + theta1),                               0.,                               0.,                            0.,                              0.,                            1 ,                               0.,                            0.],
                   [                   0.,                   0.,                      -eta2*np.cos(theta + theta2),                               0.,                               0.,                            0.,                               -1,                               0.,  -eta2*np.cos(theta + theta2),                              0.,                            0.,                               1 ,                            0.],
                   [                   0.,                   0.,                      -eta2*np.sin(theta + theta2),                               0.,                               0.,                            0.,                               0.,                               -1,  -eta2*np.sin(theta + theta2),                              0.,                            0.,                               0.,                            1 ]])
    
    fN1 = np.maximum(0,-FLOOR_SPRING_STIFFNESS*yf1 - (FLOOR_DAMPING_COEFFICIENT*yf1dot if yf1 <= 0 else 0))
    fN2 = np.maximum(0,-FLOOR_SPRING_STIFFNESS*yf2 - (FLOOR_DAMPING_COEFFICIENT*yf2dot if yf2 <= 0 else 0))
    fF1 = (-FLOOR_MU*fN1*2/np.pi*np.arctan(xf1dot*ATANGAIN/ATANDEL))
    fF2 = (-FLOOR_MU*fN2*2/np.pi*np.arctan(xf2dot*ATANGAIN/ATANDEL))
    
    C = np.matrix([[fF1 + fF2],
                   [fN1 + fN2 - (m + m1 + m2)*g],
                   [-HIP_SPRING_STIFFNESS*(phi1 - theta1 + phi2 - theta2) -HIP_DAMPING_STIFFNESS*( -theta1dot - theta2dot) + m*g*eta*np.sin(theta)],
                   [ thetadot**2*eta*np.sin(theta) + (thetadot + theta1dot)**2*gamma1*np.sin(theta + theta1)],
                   [-thetadot**2*eta*np.cos(theta) - (thetadot + theta1dot)**2*gamma1*np.cos(theta + theta1)],
                   [-m1*g*gamma1*np.sin(theta + theta1) + fN1*(l1*np.sin(theta + theta1)) + fF1*(l1*np.cos(theta + theta1)) + HIP_SPRING_STIFFNESS*(phi1 - theta1)+ HIP_DAMPING_STIFFNESS*(-theta1dot)],
                   [ thetadot**2*eta*np.sin(theta) + (thetadot + theta2dot)**2*gamma2*np.sin(theta + theta2)],
                   [-thetadot**2*eta*np.cos(theta) - (thetadot + theta2dot)**2*gamma2*np.cos(theta + theta2)],
                   [-m2*g*gamma2*np.sin(theta + theta2) + fN2*(l2*np.sin(theta + theta2)) + fF2*(l2*np.cos(theta + theta2)) + HIP_SPRING_STIFFNESS*(phi2 - theta2)+ HIP_DAMPING_STIFFNESS*(-theta2dot)],
                   [-eta1*(thetadot + theta1dot)**2*np.sin(theta + theta1)],
                   [ eta1*(thetadot + theta1dot)**2*np.cos(theta + theta1)],
                   [-eta2*(thetadot + theta2dot)**2*np.sin(theta + theta2)],
                   [ eta2*(thetadot + theta2dot)**2*np.cos(theta + theta2)]])    
    
    #test 3 - continuous simoird-like function
    #fN1 = FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf1)
    #fN2 = FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf2)
    #fF1 = (-FLOOR_MU*FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf1)*2/np.pi*np.arctan(xf1dot*ATANGAIN/ATANDEL))
    #fF2 = (-FLOOR_MU*FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf2)*2/np.pi*np.arctan(xf2dot*ATANGAIN/ATANDEL))
    
    #(np.minimum(FLOOR_MU*FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf1),np.maximum(-FLOOR_MU*FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf1), -xf1dot*FLOOR_FRICTION_STIFFNESS)))
    #fF2 = (np.minimum(FLOOR_MU*FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf2),np.maximum(-FLOOR_MU*FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf2), -xf2dot*FLOOR_FRICTION_STIFFNESS)))
     
    #test 2 - exact min/max with spring in x
    #fN1 = FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf1)
    #fN2 = FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf2)
    #fF1 = (np.minimum(FLOOR_MU*FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf1),np.maximum(-FLOOR_MU*FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf1), -xf1dot*FLOOR_FRICTION_STIFFNESS)))
    #fF2 = (np.minimum(FLOOR_MU*FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf2),np.maximum(-FLOOR_MU*FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf2), -xf2dot*FLOOR_FRICTION_STIFFNESS)))
    
    #test 1 - min/max with spring in x
    #fN1 = FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf1)
    #fN2 = FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf2)
    #fF1 = (-np.sign(xf1dot)*np.minimum(FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf1)*FLOOR_MU,np.abs(xf1dot)*FLOOR_FRICTION_STIFFNESS)) 
    #fF2 = (-np.sign(xf2dot)*np.minimum(FLOOR_SPRING_STIFFNESS*np.maximum(0,-yf2)*FLOOR_MU,np.abs(xf2dot)*FLOOR_FRICTION_STIFFNESS)) 
    
    # Calculating angular rate derivatives
    #x3dx4d = np.array(np.linalg.inv(M)*(action.reshape(2,1) - C - friction*np.matrix([[x3],[x4]]))).squeeze() 
    second_derivatives = np.array(np.linalg.inv(M)*(C)).squeeze() 
    
    # Building the derivative matrix d[state]/dt = [first_derivatives, second_derivatives]
    #print(np.concatenate((first_derivatives, second_derivatives)))
    #raise SystemExit
    derivatives = np.concatenate((first_derivatives, second_derivatives)) 

    return derivatives

# Working, pre-normal force
#    # Mass matrix
#    M = np.matrix([#                    x                    y                                                 theta                              x1                                y1                             theta1                            x2                                y2                             theta2 
#                   [                    m,                   0.,                                                0.,                               m1,                               0.,                            0.,                               m2,                               0.,                            0.],
#                   [                   0.,                   m ,                                                0.,                               0.,                               m1,                            0.,                               0.,                               m2,                            0.],
#                   [ -m*eta*np.cos(theta), -m*eta*np.sin(theta),                                                 I,                               0.,                               0.,                            0.,                               0.,                               0.,                            0.],
#                   [                  -1.,                   0., eta*np.sin(theta) + gamma1*np.sin(theta + theta1),                               1.,                               0., gamma1*np.sin(theta + theta1),                               0.,                               0.,                            0.],
#                   [                   0.,                  -1., eta*np.cos(theta) + gamma1*np.cos(theta + theta1),                               0.,                               1., gamma1*np.cos(theta + theta1),                               0.,                               0.,                            0.],
#                   [                   0.,                   0.,                                                0., m1*gamma1*np.cos(theta + theta1), m1*gamma1*np.sin(theta + theta1),                            I1,                               0.,                               0.,                            0.],
#                   [                  -1.,                   0., eta*np.sin(theta) + gamma2*np.sin(theta + theta2),                               0.,                               0.,                            0.,                               1.,                               0., gamma2*np.sin(theta + theta2)],
#                   [                   0.,                  -1., eta*np.cos(theta) + gamma2*np.cos(theta + theta2),                               0.,                               0.,                            0.,                               0.,                               1., gamma2*np.cos(theta + theta2)],
#                   [                   0.,                   0.,                                                0.,                               0.,                               0.,                            0., m2*gamma2*np.cos(theta + theta2), m2*gamma2*np.sin(theta + theta2),                            I2]])
#    
#    # C matrix
#    C = np.matrix([[fF1 + fF2],
#                   [fN1 + fN2 - (m + m1 + m2)*g],
#                   [-HIP_SPRING_STIFFNESS*(phi1 - theta1 + phi2 - theta2) + m*g*eta*np.sin(theta)],
#                   [-thetadot**2*eta*np.cos(theta) - (thetadot + theta1dot)**2*gamma1*np.cos(theta + theta1)],
#                   [ thetadot**2*eta*np.sin(theta) + (thetadot + theta1dot)**2*gamma1*np.sin(theta + theta1)],
#                   [-m1*g*gamma1*np.sin(theta+theta1) + fN1*(eta1*np.sin(theta+theta1) + gamma1*np.sin(theta+theta1)) + fF1*(eta1*np.cos(theta+theta1) + gamma1*np.cos(theta + theta1)) + HIP_SPRING_STIFFNESS*(phi1 - theta1)],
#                   [-thetadot**2*eta*np.cos(theta) - (thetadot + theta2dot)**2*gamma2*np.cos(theta+theta2)],
#                   [ thetadot**2*eta*np.sin(theta) + (thetadot + theta2dot)**2*gamma2*np.sin(theta + theta2)],
#                   [-m2*g*gamma2*np.sin(theta + theta2) + fN2*(eta2*np.sin(theta + theta2) + gamma2*np.sin(theta + theta2)) + fF2*(eta2*np.cos(theta + theta2) + gamma2*np.cos(theta + theta2)) + HIP_SPRING_STIFFNESS*(phi2 - theta2)]])    
#    



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
    
def render(filename, state_log, action_log, episode_number):
        """
        This function animates the motion of one episode. It receives the 
        log of the states encountered during one episode.
        
        Inputs: 
            state_log - a numpy array of shape [# timesteps, # states] containing all the state data
            action_log - a numpy array of shape [# timesteps] containing all the action data. The action will be an integer corresponding to which action is being performed.
                    0: No buttons pressed; 1: Q only; 2: QO; 3: QP; 4: W only; 5: WO; 6: WP; 7: O only; 8: P only
            episode_number - Which episode produced these results
            filename - Please save the animation in: 'TensorBoard/' + filename + '/videos/episode_' + str(episode_number)
            
        For reference, the state is:
            state = x, y, theta, x1, y1, theta1, x2, y2, theta2, xdot, ydot, thetadot, x1dot, y1dot, theta1dot, x2dot, y2dot, theta2dot                
        """
        
      
        
        
        # Stephane's Animating Code #
        #print(state_log[1])
        #print(action_log)
        import animator
        animator.drawState(play_game = False, filename = filename, state_log = state_log, action_log = action_log, episode_number = episode_number)
        
        #############################
        
        