
"""
This script generates the environment for reinforcement learning agents.

A full QWOP agent is modeled. It has one torso, two legs, and two arms. 
Each limb has two segments.
Arm proximal segment is link 1
Arm distal segment is link 2
Leg proximal segment is link 3
Leg distal segment is link 4

The agent is encouraged to travel down the track. Its reward is proportional
to the forward x velocity of the torso segment.

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

**************
TO IMPLEMENT
- Episode done checking
- Animation
**************
@author: Kirk Hovell (khovell@gmail.com)
"""
import numpy as np
import multiprocessing
import signal

from scipy.integrate import odeint # Numerical integrator

class Environment:
    """
    For reference, the state is:
    x,       y,    theta,    x1r,    y1r,    theta1r,    x2r,    y2r,    theta2r,    x3r,    y3r,    theta3r,    x4r,    y4r,    theta4r,    xfr,    yfr,    x1l,    y1l,    theta1l,    x2l,    y2l,    theta2l,    x3l,    y3l,    theta3l,    x4l,    y4l,    theta4l,    xfl,    yfl, \
    xdot, ydot, thetadot, x1rdot, y1rdot, theta1rdot, x2rdot, y2rdot, theta2rdot, x3rdot, y3rdot, theta3rdot, x4rdot, y4rdot, theta4rdot, xfrdot, yfrdot, x1ldot, y1ldot, theta1ldot, x2ldot, y2ldot, theta2ldot, x3ldot, y3ldot, theta3ldot, x4ldot, y4ldot, theta4ldot, xfldot, yfldot = state
    
    """
    def __init__(self): 
        ##################################
        ##### Environment Properties #####
        ##################################
        self.TOTAL_STATE_SIZE        = 62 # total number of states
        self.IRRELEVANT_STATES       = [0,3,4,6,7,9,10,12,13,15,16,17,18,20,21,23,24,26,27,29,30,34,35,37,38,40,41,43,44,46,47,48,49,51,52,54,55,57,58,60,61] # states that are irrelevant to the policy network in its decision making
        self.STATE_SIZE              = self.TOTAL_STATE_SIZE - len(self.IRRELEVANT_STATES) # total number of relevant states
        self.ACTION_SIZE             = 9
        self.TIMESTEP                = 0.05 # [s]        
        self.MAX_NUMBER_OF_TIMESTEPS = 200 # per episode
        self.NUM_FRAMES              = 100 # total animation is cut into this many frames
        self.RANDOMIZE               = False # whether or not to randomize the state & target location
        self.UPPER_STATE_BOUND       = np.array([np.inf, 4., 2*np.pi, np.inf, np.inf, 2*np.pi, np.inf, np.inf, 2*np.pi, np.inf, np.inf, 2*np.pi, np.inf, np.inf, 2*np.pi, np.inf, np.inf, np.inf, np.inf, 2*np.pi, np.inf, np.inf, 2*np.pi, np.inf, np.inf, 2*np.pi, np.inf, np.inf, 2*np.pi, np.inf, np.inf,\
                                                 np.inf, np.inf, 1., np.inf, np.inf, 1., np.inf, np.inf, 1., np.inf, np.inf, 1., np.inf, np.inf, 1., np.inf, np.inf, np.inf, np.inf, 1., np.inf, np.inf, 1., np.inf, np.inf, 1., np.inf, np.inf, 1., np.inf, np.inf])
        self.NORMALIZE_STATE         = True # Normalize state on each timestep to avoid vanishing gradients
        self.MIN_Q                   = -5000.0
        self.MAX_Q                   = 15000.0
        self.DONE_ON_FALL            = True # whether or not falling down ends the episode
        
        # Rendering parameters
        self.HEIGHT = 500
        self.WIDTH  = 800
        self.HUMAN_SCALE = 150 #pixel/m
        
        self.x_0 = np.rint(self.WIDTH/2) # 0 in x on the screen
        self.y_0 = np.rint(self.HEIGHT*9/10) # 0 in y on the screen
        
        # Initial Conditions
        self.INITIAL_Y       =   2.          # [m]
        self.INITIAL_X       =   0.          # [m]
        self.INITIAL_THETA   =   0*np.pi/180 # [rad]
        self.INITIAL_THETA1R =  30*np.pi/180 # [rad]
        self.INITIAL_THETA2R =  30*np.pi/180 # [rad]
        self.INITIAL_THETA3R =  30*np.pi/180 # [rad]
        self.INITIAL_THETA4R = -10*np.pi/180 # [rad]
        self.INITIAL_THETA1L = -30*np.pi/180 # [rad]
        self.INITIAL_THETA2L =  30*np.pi/180 # [rad]
        self.INITIAL_THETA3L = -20*np.pi/180 # [rad]
        self.INITIAL_THETA4L = -10*np.pi/180 # [rad]        
        
        # How much the leg desired angle changes per frame when a button is pressed
        self.HIP_INCREMENT      =   5*np.pi/180 # [rad/s]
        self.CALF_INCREMENT     =   5*np.pi/180 # [rad/s]
        self.SHOULDER_INCREMENT =   5*np.pi/180 # [rad/s]
        self.ELBOW_INCREMENT    =   5*np.pi/180 # [rad/s]
        self.PHI1R_INITIAL       =  30*np.pi/180 # [rad]
        self.PHI2R_INITIAL       =  30*np.pi/180 # [rad]
        self.PHI3R_INITIAL       =  30*np.pi/180 # [rad]
        self.PHI4R_INITIAL       = -10*np.pi/180 # [rad]
        self.PHI1L_INITIAL       = -30*np.pi/180 # [rad]
        self.PHI2L_INITIAL       =  30*np.pi/180 # [rad]
        self.PHI3L_INITIAL       = -20*np.pi/180 # [rad]
        self.PHI4L_INITIAL       = -10*np.pi/180 # [rad]
        
        # Joint springs and dampers
        self.HIP_SPRING_STIFFNESS      = 10000 # [Nm/rad]       
        self.HIP_DAMPING               =   100 # [Nms/rad]
        self.CALF_SPRING_STIFFNESS     = 10000 # [Nm/rad]
        self.CALF_DAMPING              =   100 # [Nms/rad]
        self.SHOULDER_SPRING_STIFFNESS = 10000 # [Nm/rad]
        self.SHOULDER_DAMPING          =   100 # [Nms/rad]
        self.ELBOW_SPRING_STIFFNESS    = 10000 # [Nm/rad]
        self.ELBOW_DAMPING             =   100 # [Nms/rad]
        
        # Friction properties
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
        #self.SEGMENT_ETA_LENGTH = list()# [m]
        self.SEGMENT_GAMMA_LENGTH = list()# [m]
        self.SEGMENT_PHI_NAUGTH = list()# [rad]
        self.SEGMENT_PHI_MAX = list()# [rad]
        self.SEGMENT_PHI_MIN = list()# [rad]
        
        
        #player stats
        p_mass = 75#kg
        p_height = 1.80#m

#lengths from page 14 Conti http://www.oandplibrary.org/al/pdf/1972_01_001.pdf
#gamma from page 11, NYU for completeness
#radius of gyration from table 10, page 12, weighted average   
        
        
        # segment 0, main body (+ head and neck contributions)
        m = (0.497+0.181)*p_mass
        l = 0.289*p_height
        gamma = 0.5*l
        #phi0 = 0 
        phi_min = -1000000
        phi_max =  1000000
        k = 0.5*l
        I = (0.497*p_mass)*k**2+(0.181*p_mass)*(0.495*0.181*p_height)**2+(0.181*p_mass)*((0.567*0.181*p_height)+gamma)**2
        
        
        self.HEAD_DIAM = 0.13*p_height
        self.NECK_LENGTH = 0.116*p_height
        
        self.SEGMENT_MASS.append(m)
        self.SEGMENT_MOMINERT.append(I)
        self.SEGMENT_LENGTH.append(l)
        self.SEGMENT_GAMMA_LENGTH.append(gamma)
        #self.SEGMENT_PHI_NAUGTH.append(phi0)
        self.SEGMENT_PHI_MAX.append(phi_max)
        self.SEGMENT_PHI_MIN.append(phi_min)
        
        
        # segment 1, proximal arm
        m = 0.028*p_mass
        l = 0.189*p_height
        gamma = 0.449*l
        #phi0 = -np.pi/6 
        phi_min = -np.pi/4
        phi_max =  np.pi/4
        k = 0.268*l
        I =  k**2*m
       
        self.SEGMENT_MASS.append(m)
        self.SEGMENT_MOMINERT.append(I)
        self.SEGMENT_LENGTH.append(l)
        self.SEGMENT_GAMMA_LENGTH.append(gamma)
        #self.SEGMENT_PHI_NAUGTH.append(phi0)
        self.SEGMENT_PHI_MAX.append(phi_max)
        self.SEGMENT_PHI_MIN.append(phi_min)
              
        
        # segment 2, distal arm (+hand)
        m = 0.0465*p_mass+0.100*p_mass
        l = 0.273*p_height
        gamma = 0.382*l
        #phi0 = np.pi/2 
        phi_min = 0
        phi_max =  5*np.pi/6
        k = 0.263*l
        I =  k**2*m
        
        self.SEGMENT_MASS.append(m)
        self.SEGMENT_MOMINERT.append(I)
        self.SEGMENT_LENGTH.append(l)
        self.SEGMENT_GAMMA_LENGTH.append(gamma)
        #self.SEGMENT_PHI_NAUGTH.append(phi0)
        self.SEGMENT_PHI_MAX.append(phi_max)
        self.SEGMENT_PHI_MIN.append(phi_min)
        

        # segment 3, proximal leg
        m = 0.0465*p_mass+0.100*p_mass
        l = 0.245*p_height
        gamma = 0.41*l
        #phi0 = -np.pi/6 
        phi_min = -np.pi/2
        phi_max =  0
        k = 0.25*l
        I = k**2*m
        
        self.SEGMENT_MASS.append(m)
        self.SEGMENT_MOMINERT.append(I)
        self.SEGMENT_LENGTH.append(l)
        self.SEGMENT_GAMMA_LENGTH.append(gamma)
        #self.SEGMENT_PHI_NAUGTH.append(phi0)
        self.SEGMENT_PHI_MAX.append(phi_max)
        self.SEGMENT_PHI_MIN.append(phi_min)
        
        # segment 4, distal leg (+foot)
        m = 0.0465*p_mass+0.100*p_mass
        l = 0.328*p_height
        gamma = 0.45*l
        #phi0 = -np.pi/6 
        phi_min = -np.pi/4
        phi_max =  np.pi/2
        k = 0.303*l
        I = k**2*m
        
        self.SEGMENT_MASS.append(m)
        self.SEGMENT_MOMINERT.append(I)
        self.SEGMENT_LENGTH.append(l)
        self.SEGMENT_GAMMA_LENGTH.append(gamma)
        #self.SEGMENT_PHI_NAUGTH.append(phi0)
        self.SEGMENT_PHI_MAX.append(phi_max)
        self.SEGMENT_PHI_MIN.append(phi_min)
        
        

                
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
            # Randomizing initial conditions for each episode
            pass # to be completed later
            
        else:
            # Constant initial conditions on each episode
            
            # Torso
            initial_x = self.INITIAL_X
            initial_y = self.INITIAL_Y
            initial_theta = self.INITIAL_THETA
            
            # Right proximal arm segment
            initial_x1r = initial_x - self.SEGMENT_GAMMA_LENGTH[0] * np.sin(initial_theta) + self.SEGMENT_GAMMA_LENGTH[1] * np.sin(self.INITIAL_THETA1R)
            initial_y1r = initial_y + self.SEGMENT_GAMMA_LENGTH[0] * np.cos(initial_theta) - self.SEGMENT_GAMMA_LENGTH[1] * np.cos(self.INITIAL_THETA1R)
            initial_theta1r = self.INITIAL_THETA1R
            
            # Right distal arm segment
            initial_x2r = initial_x1r + (self.SEGMENT_LENGTH[1] - self.SEGMENT_GAMMA_LENGTH[1]) * np.sin(self.INITIAL_THETA1R) + self.SEGMENT_GAMMA_LENGTH[2] * np.sin(self.INITIAL_THETA2R)
            initial_y2r = initial_y1r - (self.SEGMENT_LENGTH[1] - self.SEGMENT_GAMMA_LENGTH[1]) * np.cos(self.INITIAL_THETA1R) - self.SEGMENT_GAMMA_LENGTH[2] * np.cos(self.INITIAL_THETA2R)
            initial_theta2r = self.INITIAL_THETA2R
            
            # Left proximal arm segment
            initial_x1l = initial_x - self.SEGMENT_GAMMA_LENGTH[0] * np.sin(initial_theta) - self.SEGMENT_GAMMA_LENGTH[1] * np.sin(self.INITIAL_THETA1L)
            initial_y1l = initial_y + self.SEGMENT_GAMMA_LENGTH[0] * np.cos(initial_theta) - self.SEGMENT_GAMMA_LENGTH[1] * np.cos(self.INITIAL_THETA1L)
            initial_theta1l = self.INITIAL_THETA1L
            
            # Left distal arm segment
            initial_x2l = initial_x1l - (self.SEGMENT_LENGTH[1] - self.SEGMENT_GAMMA_LENGTH[1]) * np.sin(self.INITIAL_THETA1L) - self.SEGMENT_GAMMA_LENGTH[2] * np.sin(self.INITIAL_THETA2L)
            initial_y2l = initial_y1l - (self.SEGMENT_LENGTH[1] - self.SEGMENT_GAMMA_LENGTH[1]) * np.cos(self.INITIAL_THETA1L) - self.SEGMENT_GAMMA_LENGTH[2] * np.cos(self.INITIAL_THETA2L)
            initial_theta2l = self.INITIAL_THETA2L   
            
            # Right proximal leg segment
            initial_x3r = initial_x + (self.SEGMENT_LENGTH[0] - self.SEGMENT_GAMMA_LENGTH[0]) * np.sin(initial_theta) + self.SEGMENT_GAMMA_LENGTH[3] * np.sin(self.INITIAL_THETA3R)
            initial_y3r = initial_y - (self.SEGMENT_LENGTH[0] - self.SEGMENT_GAMMA_LENGTH[0]) * np.cos(initial_theta) - self.SEGMENT_GAMMA_LENGTH[3] * np.cos(self.INITIAL_THETA3R)
            initial_theta3r = self.INITIAL_THETA3R
            
            # Right distal leg segment
            initial_x4r = initial_x3r + (self.SEGMENT_LENGTH[3] - self.SEGMENT_GAMMA_LENGTH[3]) * np.sin(initial_theta3r) + self.SEGMENT_GAMMA_LENGTH[4] * np.sin(self.INITIAL_THETA4R)
            initial_y4r = initial_y3r - (self.SEGMENT_LENGTH[3] - self.SEGMENT_GAMMA_LENGTH[3]) * np.cos(initial_theta3r) - self.SEGMENT_GAMMA_LENGTH[4] * np.cos(self.INITIAL_THETA4R)
            initial_theta4r = self.INITIAL_THETA4R
            
            # Right foot
            initial_xfr = initial_x4r + (self.SEGMENT_LENGTH[4] - self.SEGMENT_GAMMA_LENGTH[4]) * np.sin(initial_theta4r)
            initial_yfr = initial_y4r - (self.SEGMENT_LENGTH[4] - self.SEGMENT_GAMMA_LENGTH[4]) * np.cos(initial_theta4r)
            
            # Left proximal leg segment
            initial_x3l = initial_x + (self.SEGMENT_LENGTH[0] - self.SEGMENT_GAMMA_LENGTH[0]) * np.sin(initial_theta) - self.SEGMENT_GAMMA_LENGTH[3] * np.sin(self.INITIAL_THETA3L)
            initial_y3l = initial_y - (self.SEGMENT_LENGTH[0] - self.SEGMENT_GAMMA_LENGTH[0]) * np.cos(initial_theta) - self.SEGMENT_GAMMA_LENGTH[3] * np.cos(self.INITIAL_THETA3L)
            initial_theta3l = self.INITIAL_THETA3L
            
            # Left distal leg segment
            initial_x4l = initial_x3l - (self.SEGMENT_LENGTH[3] - self.SEGMENT_GAMMA_LENGTH[3]) * np.sin(initial_theta3l) - self.SEGMENT_GAMMA_LENGTH[4] * np.sin(self.INITIAL_THETA4L)
            initial_y4l = initial_y3l - (self.SEGMENT_LENGTH[3] - self.SEGMENT_GAMMA_LENGTH[3]) * np.cos(initial_theta3l) - self.SEGMENT_GAMMA_LENGTH[4] * np.cos(self.INITIAL_THETA4L)
            initial_theta4l = self.INITIAL_THETA4L
            
            # Left foot
            initial_xfl = initial_x4l - (self.SEGMENT_LENGTH[4] - self.SEGMENT_GAMMA_LENGTH[4]) * np.sin(initial_theta4l)
            initial_yfl = initial_y4l - (self.SEGMENT_LENGTH[4] - self.SEGMENT_GAMMA_LENGTH[4]) * np.cos(initial_theta4l)
            
            # Resetting joint angle setpoints
            self.phi1r = self.PHI1L_INITIAL 
            self.phi2r = self.PHI2L_INITIAL 
            self.phi3r = self.PHI3L_INITIAL
            self.phi4r = self.PHI4L_INITIAL
            self.phi1l = self.PHI1L_INITIAL 
            self.phi2l = self.PHI2L_INITIAL 
            self.phi3l = self.PHI3L_INITIAL
            self.phi4l = self.PHI4L_INITIAL
            
            # Assembling into the state
            self.state = np.array([  initial_x,   initial_y,   initial_theta,\
                                   initial_x1r, initial_y1r, initial_theta1r,\
                                   initial_x2r, initial_y2r, initial_theta2r,\
                                   initial_x3r, initial_y3r, initial_theta3r,\
                                   initial_x4r, initial_y4r, initial_theta4r,\
                                   initial_xfr, initial_yfr, \
                                   initial_x1l, initial_y1l, initial_theta1l,\
                                   initial_x2l, initial_y2l, initial_theta2l,\
                                   initial_x3l, initial_y3l, initial_theta3l,\
                                   initial_x4l, initial_y4l, initial_theta4l,\
                                   initial_xfl, initial_yfl, \
                                            0.,          0.,              0.,\
                                            0.,          0.,              0.,\
                                            0.,          0.,              0.,\
                                            0.,          0.,              0.,\
                                            0.,          0.,              0.,\
                                            0.,          0., \
                                            0.,          0.,              0.,\
                                            0.,          0.,              0.,\
                                            0.,          0.,              0.,\
                                            0.,          0.,              0.,\
                                            0.,          0.                    ])
        """
        For reference, the state is:
        x,       y,    theta,    x1r,    y1r,    theta1r,    x2r,    y2r,    theta2r,    x3r,    y3r,    theta3r,    x4r,    y4r,    theta4r,    xfr,    yfr,    x1l,    y1l,    theta1l,    x2l,    y2l,    theta2l,    x3l,    y3l,    theta3l,    x4l,    y4l,    theta4l,    xfl,    yfl, \
        xdot, ydot, thetadot, x1rdot, y1rdot, theta1rdot, x2rdot, y2rdot, theta2rdot, x3rdot, y3rdot, theta3rdot, x4rdot, y4rdot, theta4rdot, xfrdot, yfrdot, x1ldot, y1ldot, theta1ldot, x2ldot, y2ldot, theta2ldot, x3ldot, y3ldot, theta3ldot, x4ldot, y4ldot, theta4ldot, xfldot, yfldot = state
        
        """
        
        # Resetting the time
        self.time = 0.0  
        
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
        #segment_points[0,:,:] = self.returnPointCoords(x,y,theta,self.SEGMENT_GAMMA_LENGTH[0],self.SEGMENT_ETA_LENGTH[0],x,self.x_0,self.y_0,self.HUMAN_SCALE)
        #segment_points[1,:,:] = self.returnPointCoords(x1,y1,theta+theta1,self.SEGMENT_GAMMA_LENGTH[1],self.SEGMENT_ETA_LENGTH[1],x,self.x_0,self.y_0,self.HUMAN_SCALE)
        #segment_points[2,:,:] = self.returnPointCoords(x2,y2,theta+theta2,self.SEGMENT_GAMMA_LENGTH[2],self.SEGMENT_ETA_LENGTH[2],x,self.x_0,self.y_0,self.HUMAN_SCALE)
        
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
        # Q and W control the shoulders and hips through phi1 and phi3
        # O and P control the elbows and calves through phi2 and phi4
        if pressed_q:
            self.phi1r += self.SHOULDER_INCREMENT
            self.phi1l -= self.SHOULDER_INCREMENT
            self.phi3r -= self.HIP_INCREMENT
            self.phi3l += self.HIP_INCREMENT
        if pressed_w:
            self.phi1r -= self.SHOULDER_INCREMENT
            self.phi1l += self.SHOULDER_INCREMENT
            self.phi3r += self.HIP_INCREMENT
            self.phi3l -= self.HIP_INCREMENT
        if pressed_o:
            self.phi2r += self.ELBOW_INCREMENT
            self.phi2l -= self.ELBOW_INCREMENT
            self.phi4r -= self.CALF_INCREMENT
            self.phi4l += self.CALF_INCREMENT
        if pressed_p:
            self.phi2r -= self.ELBOW_INCREMENT
            self.phi2l += self.ELBOW_INCREMENT
            self.phi4r += self.CALF_INCREMENT
            self.phi4l -= self.CALF_INCREMENT 
        
        # Packing up the parameters the equations of motion need
        parameters = np.array([self.SEGMENT_MASS[0], self.SEGMENT_MASS[1], self.SEGMENT_MASS[2], self.SEGMENT_MASS[3], self.SEGMENT_MASS[4], \
                               self.SEGMENT_LENGTH[0], self.SEGMENT_LENGTH[1], self.SEGMENT_LENGTH[2], self.SEGMENT_LENGTH[3], self.SEGMENT_LENGTH[4], \
                               self.SEGMENT_GAMMA_LENGTH[0], self.SEGMENT_GAMMA_LENGTH[1], self.SEGMENT_GAMMA_LENGTH[2], self.SEGMENT_GAMMA_LENGTH[3], self.SEGMENT_GAMMA_LENGTH[4], \
                               self.SEGMENT_MOMINERT[0], self.SEGMENT_MOMINERT[1], self.SEGMENT_MOMINERT[2], self.SEGMENT_MOMINERT[3], self.SEGMENT_MOMINERT[4], \
                               self.HIP_SPRING_STIFFNESS, self.HIP_DAMPING, self.CALF_SPRING_STIFFNESS, self.CALF_DAMPING, \
                               self.SHOULDER_SPRING_STIFFNESS, self.SHOULDER_DAMPING, self.ELBOW_SPRING_STIFFNESS, self.ELBOW_DAMPING, \
                               self.g, self.FLOOR_SPRING_STIFFNESS, self.FLOOR_DAMPING_COEFFICIENT, self.FLOOR_FRICTION_STIFFNESS, self.FLOOR_MU, self.ATANDEL, self.ATANGAIN, \
                               self.phi1r, self.phi2r, self.phi3r, self.phi4r, self.phi1l, self.phi2l, self.phi3l, self.phi4l], dtype = 'float64')
    
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
    x,       y,    theta,    x1r,    y1r,    theta1r,    x2r,    y2r,    theta2r,    x3r,    y3r,    theta3r,    x4r,    y4r,    theta4r,    xfr,    yfr,    x1l,    y1l,    theta1l,    x2l,    y2l,    theta2l,    x3l,    y3l,    theta3l,    x4l,    y4l,    theta4l,    xfl,    yfl, \
    xdot, ydot, thetadot, x1rdot, y1rdot, theta1rdot, x2rdot, y2rdot, theta2rdot, x3rdot, y3rdot, theta3rdot, x4rdot, y4rdot, theta4rdot, xfrdot, yfrdot, x1ldot, y1ldot, theta1ldot, x2ldot, y2ldot, theta2ldot, x3ldot, y3ldot, theta3ldot, x4ldot, y4ldot, theta4ldot, xfldot, yfldot = state
    
    # Unpacking parameters
    m, m1, m2, m3, m4, \
    l, l1, l2, l3, l4, \
    gamma, gamma1, gamma2, gamma3, gamma4,\
    I, I1, I2, I3, I4, \
    k3, c3, k4, c4, \
    k1, c1, k2, c2, \
    g, FLOOR_SPRING_STIFFNESS, FLOOR_DAMPING_COEFFICIENT, FLOOR_FRICTION_STIFFNESS, FLOOR_MU, ATANDEL, ATANGAIN, \
    phi1r, phi2r, phi3r, phi4r, phi1l, phi2l, phi3l, phi4l = parameters 
    
    first_derivatives = np.array([xdot, ydot, thetadot, x1rdot, y1rdot, theta1rdot, x2rdot, y2rdot, theta2rdot, x3rdot, y3rdot, theta3rdot, x4rdot, y4rdot, theta4rdot, xfrdot, yfrdot, x1ldot, y1ldot, theta1ldot, x2ldot, y2ldot, theta2ldot, x3ldot, y3ldot, theta3ldot, x4ldot, y4ldot, theta4ldot, xfldot, yfldot]) 
    
    M = np.matrix([
                    [m	,	0	,	0	,	m1	,	0	,	0	,	m2	,	0	,	0	,	m3	,	0	,	0	,	m4	,	0	,	0	,	0	,	0	,	m1	,	0	,	0	,	m2	,	0	,	0	,	m3	,	0	,	0	,	m4	,	0	,	0	,	0	,	0	],
                    [	0	,	m	,	0	,	0	,	m1	,	0	,	0	,	m2	,	0	,	0	,	m3	,	0	,	0	,	m4	,	0	,	0	,	0	,	0	,	m1	,	0	,	0	,	m2	,	0	,	0	,	m3	,	0	,	0	,	m4	,	0	,	0	,	0	],
                    [	0	,	0	,	I	,	-m1*gamma*np.cos(theta)	,	-m1*gamma*np.sin(theta)	,	0	,	-m2*gamma*np.cos(theta)	,	-m2*gamma*np.sin(theta)	,	0	,	m3*(l-gamma)*np.cos(theta)	,	m3*(l-gamma)*np.sin(theta)	,	0	,	m4*(l-gamma)*np.cos(theta)	,	m4*(l-gamma)*np.sin(theta)	,	0	,	0	,	0	,	-m1*gamma*np.cos(theta)	,	-m1*gamma*np.sin(theta)	,	0	,	-m2*gamma*np.cos(theta)	,	-m2*gamma*np.sin(theta)	,	0	,	m3*(l-gamma)*np.cos(theta)	,	m3*(l-gamma)*np.sin(theta)	,	0	,	m4*(l-gamma)*np.cos(theta)	,	m4*(l-gamma)*np.sin(theta)	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	m1*gamma1*np.cos(theta1r)	,	m1*gamma1*np.sin(theta1r)	,	I1	,	m2*gamma1*np.cos(theta1r)+m2*(l1-gamma1)*np.cos(theta1r)	,	m2*gamma1*np.sin(theta1r)+m2*(l1-gamma1)*np.sin(theta1r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	m2*gamma2*np.cos(theta2r)	,	m2*gamma2*np.sin(theta2r)	,	I2	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m3*gamma3*np.cos(theta3r)	,	m3*gamma3*np.sin(theta3r)	,	I3	,	m4*gamma3*np.cos(theta3r)+m4*(l3-gamma3)*np.cos(theta3r)	,	m4*gamma3*np.sin(theta3r)+m4*(l3-gamma3)*np.sin(theta3r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m4*gamma4*np.cos(theta4r)	,	m4*gamma4*np.sin(theta4r)	,	I4	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	-1	,	0	,	gamma*np.cos(theta)	,	1	,	0	,	-gamma1*np.cos(theta1r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	-1	,	gamma*np.sin(theta)	,	0	,	1	,	-gamma1*np.sin(theta1r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	-1	,	0	,	-(l1-gamma1)*np.cos(theta1r)	,	1	,	0	,	-gamma2*np.cos(theta2r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	-1	,	-(l1-gamma1)*np.sin(theta1r)	,	0	,	1	,	-gamma2*np.sin(theta2r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	0	,	-(l-gamma)*np.cos(theta)	,	1	,	0	,	-gamma3*np.cos(theta3r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	-(l-gamma)*np.sin(theta)	,	0	,	1	,	-gamma3*np.sin(theta3r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	0	,	-(l3-gamma3)*np.cos(theta3r)	,	1	,	0	,	-gamma4*np.cos(theta4r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	-(l3-gamma3)*np.sin(theta3r)	,	0	,	1	,	-gamma4*np.sin(theta4r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	0	,	-(l4-gamma4)*np.cos(theta4r)	,	1	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	-(l4-gamma4)*np.sin(theta4r)	,	0	,	1	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m1*gamma1*np.cos(theta1l)	,	m1*gamma1*np.sin(theta1l)	,	I1	,	m2*gamma1*np.cos(theta1l)+m2*(l1-gamma1)*np.cos(theta1l)	,	m2*gamma1*np.sin(theta1l)+m2*(l1-gamma1)*np.sin(theta1l)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m2*gamma2*np.cos(theta2l)	,	m2*gamma2*np.sin(theta2l)	,	I2	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m3*gamma3*np.cos(theta3l)	,	m3*gamma3*np.sin(theta3l)	,	I3	,	m4*gamma3*np.cos(theta3l)+m4*(l3-gamma3)*np.cos(theta3l)	,	m4*gamma3*np.sin(theta3l)+m4*(l3-gamma3)*np.sin(theta3l)	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m4*gamma4*np.cos(theta4l)	,	m4*gamma4*np.sin(theta4l)	,	I4	,	0	,	0	],
                    [	-1	,	0	,	gamma*np.cos(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	0	,	-gamma1*np.cos(theta1l)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	-1	,	gamma*np.sin(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	-gamma1*np.sin(theta1l)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	0	,	-(l1-gamma1)*np.cos(theta1l)	,	1	,	0	,	-gamma2*np.cos(theta2l)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	-(l1-gamma1)*np.sin(theta1l)	,	0	,	1	,	-gamma2*np.sin(theta2l)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	0	,	-(l-gamma)*np.cos(theta)	,	1	,	0	,	-gamma3*np.cos(theta3l)	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	-(l-gamma)*np.sin(theta)	,	0	,	1	,	-gamma3*np.sin(theta3l)	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	0	,	-(l3-gamma3)*np.cos(theta3l)	,	1	,	0	,	-gamma4*np.cos(theta4l)	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	-(l3-gamma3)*np.sin(theta3l)	,	0	,	1	,	-gamma4*np.sin(theta4l)	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	0	,	-(l4-gamma4)*np.cos(theta4l)	,	1	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	-(l4-gamma4)*np.sin(theta4l)	,	0	,	1	]])
    
    # Calculating floor reaction forces
    fNr = np.maximum(0,-FLOOR_SPRING_STIFFNESS*yfr - (FLOOR_DAMPING_COEFFICIENT*yfrdot if yfr <= 0 else 0))
    fNl = np.maximum(0,-FLOOR_SPRING_STIFFNESS*yfl - (FLOOR_DAMPING_COEFFICIENT*yfldot if yfl <= 0 else 0))
    fFr = (-FLOOR_MU*fNr*2/np.pi*np.arctan(xfrdot*ATANGAIN/ATANDEL))
    fFl = (-FLOOR_MU*fNl*2/np.pi*np.arctan(xfldot*ATANGAIN/ATANDEL))
    
    C = np.matrix([ [	fFr+fFl	],
                    [	-m*g-2*m1*g-2*m2*g-2*m3*g-2*m4*g+fNr+fNl	],
                    [	2*m1*g*gamma*np.sin(theta)+2*m2*g*gamma*np.sin(theta)+fFr*(l-gamma)*np.cos(theta)+fFl*(l-gamma)*np.cos(theta)-2*m3*g*(l-gamma)*np.sin(theta)-2*m4*g*(l-gamma)*np.sin(theta)+fNr*(l-gamma)*np.sin(theta)+fNl*(l-gamma)*np.sin(theta)-k1*(theta+phi1l-theta1l)-c1*(thetadot-theta1ldot)-k1*(theta+phi1r-theta1r)-c1*(thetadot-theta1rdot)-k3*(theta+phi3l-theta3l)-c3*(thetadot-theta3ldot)-k3*(theta+phi3r-theta3r)-c3*(thetadot-theta3rdot)	],
                    [	-m1*g*gamma1*np.sin(theta1r)-m2*g*gamma1*np.sin(theta1r)-m2*g*(l1-gamma1)*np.sin(theta1r)-k2*(theta1r+phi2r-theta2r)-c2*(theta1rdot-theta2rdot)+k1*(theta+phi1r-theta1r)+c1*(thetadot-theta1rdot)	],
                    [	-m2*g*gamma2*np.sin(theta2r)+k2*(theta1r+phi2r-theta2r)+c2*(theta1rdot-theta2rdot)	],
                    [	-m3*g*gamma3*np.sin(theta3r)-m4*g*gamma3*np.sin(theta3r)-m4*g*(l3-gamma3)*np.sin(theta3r)-k4*(theta3r+phi4r-theta4r)-c4*(theta3rdot-theta4rdot)+k3*(theta+phi3r-theta3r)+c3*(thetadot-theta3rdot)+fFr*gamma3*np.cos(theta3r)+fNr*gamma3*np.sin(theta3r)+fFr*(l3-gamma3)*np.cos(theta3r)+fNr*(l3-gamma3)*np.sin(theta3r)	],
                    [	-m4*g*gamma4*np.sin(theta4r)+k4*(theta3r+phi4r-theta4r)+c4*(theta3rdot-theta4rdot)+fFr*gamma4*np.cos(theta4r)+fNr*gamma4*np.sin(theta4r)+fFr*(l4-gamma4)*np.cos(theta4r)+fNr*(l4-gamma4)*np.sin(theta4r)	],
                    [	gamma*thetadot**2*np.sin(theta)-gamma1*theta1rdot**2*np.sin(theta1r)	],
                    [	-gamma*thetadot**2*np.cos(theta)+gamma1*theta1rdot**2*np.cos(theta1r)	],
                    [	-(l1-gamma1)*theta1rdot**2*np.sin(theta1r)-gamma2*theta2rdot**2*np.sin(theta2r)	],
                    [	(l1-gamma1)*theta1rdot**2*np.cos(theta1r)+gamma2*theta2rdot**2*np.cos(theta2r)	],
                    [	-(l-gamma)*thetadot**2*np.sin(theta)-gamma3*theta3rdot**2*np.sin(theta3r)	],
                    [	(l-gamma)*thetadot**2*np.cos(theta)+gamma3*theta3rdot**2*np.cos(theta3r)	],
                    [	-(l3-gamma3)*theta3rdot**2*np.sin(theta3r)-gamma4*theta4rdot**2*np.sin(theta4r)	],
                    [	(l3-gamma3)*theta3rdot**2*np.cos(theta3r)+gamma4*theta4rdot**2*np.cos(theta4r)	],
                    [	-(l4-gamma4)*theta4rdot**2*np.sin(theta4r)	],
                    [	(l4-gamma4)*theta4rdot**2*np.cos(theta4r)	],
                    [	-m1*g*gamma1*np.sin(theta1l)-m2*g*gamma1*np.sin(theta1l)-m2*g*(l1-gamma1)*np.sin(theta1l)-k2*(theta1l+phi2l-theta2l)-c2*(theta1ldot-theta2ldot)+k1*(theta+phi1l-theta1l)+c1*(thetadot-theta1ldot)	],
                    [	-m2*g*gamma2*np.sin(theta2l)+k2*(theta1l+phi2l-theta2l)+c2*(theta1ldot-theta2ldot)	],
                    [	-m3*g*gamma3*np.sin(theta3l)-m4*g*gamma3*np.sin(theta3l)-m4*g*(l3-gamma3)*np.sin(theta3l)-k4*(theta3l+phi4l-theta4l)-c4*(theta3ldot-theta4ldot)+k3*(theta+phi3l-theta3l)+c3*(thetadot-theta3ldot)+fFl*gamma3*np.cos(theta3l)+fNl*gamma3*np.sin(theta3l)+fFl*(l3-gamma3)*np.cos(theta3l)+fNl*(l3-gamma3)*np.sin(theta3l)	],
                    [	-m4*g*gamma4*np.sin(theta4l)+k4*(theta3l+phi4l-theta4l)+c4*(theta3ldot-theta4ldot)+fFl*gamma4*np.cos(theta4l)+fNl*gamma4*np.sin(theta4l)+fFl*(l4-gamma4)*np.cos(theta4l)+fNl*(l4-gamma4)*np.sin(theta4l)	],
                    [	gamma*thetadot**2*np.sin(theta)-gamma1*theta1ldot**2*np.sin(theta1l)	],
                    [	-gamma*thetadot**2*np.cos(theta)+gamma1*theta1ldot**2*np.cos(theta1l)	],
                    [	-(l1-gamma1)*theta1ldot**2*np.sin(theta1l)-gamma2*theta2ldot**2*np.sin(theta2l)	],
                    [	(l1-gamma1)*theta1ldot**2*np.cos(theta1l)+gamma2*theta2ldot**2*np.cos(theta2l)	],
                    [	-(l-gamma)*thetadot**2*np.sin(theta)-gamma3*theta3ldot**2*np.sin(theta3l)	],
                    [	(l-gamma)*thetadot**2*np.cos(theta)+gamma3*theta3ldot**2*np.cos(theta3l)	],
                    [	-(l3-gamma3)*theta3ldot**2*np.sin(theta3l)-gamma4*theta4ldot**2*np.sin(theta4l)	],
                    [	(l3-gamma3)*theta3ldot**2*np.cos(theta3l)+gamma4*theta4ldot**2*np.cos(theta4l)	],
                    [	-(l4-gamma4)*theta4ldot**2*np.sin(theta4l)	],
                    [	(l4-gamma4)*theta4ldot**2*np.cos(theta4l)	]])

    # Calculating second derivatives
    second_derivatives = np.array(np.linalg.inv(M)*(C)).squeeze() 
    
    # Building the derivative matrix d[state]/dt = [first_derivatives, second_derivatives]
    #print(np.concatenate((first_derivatives, second_derivatives)))
    #raise SystemExit
    derivatives = np.concatenate((first_derivatives, second_derivatives)) 

    return derivatives

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
        x,       y,    theta,    x1r,    y1r,    theta1r,    x2r,    y2r,    theta2r,    x3r,    y3r,    theta3r,    x4r,    y4r,    theta4r,    xfr,    yfr,    x1l,    y1l,    theta1l,    x2l,    y2l,    theta2l,    x3l,    y3l,    theta3l,    x4l,    y4l,    theta4l,    xfl,    yfl, \
        xdot, ydot, thetadot, x1rdot, y1rdot, theta1rdot, x2rdot, y2rdot, theta2rdot, x3rdot, y3rdot, theta3rdot, x4rdot, y4rdot, theta4rdot, xfrdot, yfrdot, x1ldot, y1ldot, theta1ldot, x2ldot, y2ldot, theta2ldot, x3ldot, y3ldot, theta3ldot, x4ldot, y4ldot, theta4ldot, xfldot, yfldot = state
        """
        
        
      
        
        
        # Stephane's Animating Code #
        #print(state_log[1])
        #print(action_log)
        import animator_full
        animator_full.drawState(play_game = False, filename = filename, state_log = state_log, action_log = action_log, episode_number = episode_number)
        
        #############################
        
        