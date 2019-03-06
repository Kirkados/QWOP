
"""
This script generates the environment for reinforcement learning agents.

OpenAI's gym environments are used here to test the performance of the C51
learning algorithm.

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
#import animator
#import pygame
#from scipy.integrate import odeint # Numerical integrator

class Environment:
    
    def __init__(self): 
        ##################################
        ##### Environment Properties #####
        ##################################
        self.state_size = 18
        self.action_size = 2
        self.TIMESTEP = 0.1 # [s]
        self.target_reward = 1.
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
        self.time = 0.0  
        
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
        parameters = np.array([self.m, self.m1, self.m2, self.eta, self.eta1, self.eta2, self.gamma1, self.gamma2, self.I, self.I1, self.I2, self.g, fF1, fF2, self.phi1, self.phi2, fN1, fN2, K], dtype = 'float64')

        # Integrating forward one time step. 
        # Returns initial condition on first row then next timestep on the next row
        ##############################
        ##### PROPAGATE DYNAMICS #####
        ##############################
        next_states = odeint(equations_of_motion, self.state, [self.time, self.time + self.timestep], args = (parameters,), full_output = 0)
        
        reward = self.reward_function(action) 
        
        self.state = next_states[1,:] # remembering the current state
        self.time += self.timestep # updating the stored time
        

        # Return the (state, reward, done)
        return self.state, reward, done


    def reward_function(self, action):
        # Returns the reward for this timestep as a function of the state and action
        
        # The agent is (currently) rewarded for forward velocity.
        reward = self.state[9]        
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
    
def render(state_log, action_log, time_log, episode_number, filename):
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
        print(state_log)
        print(action_log)
        print(time_log)
        
        #############################
        
                
        #initialize the pygame window
        width = 800
        height = 500
        size = [width, height]
        pygame.init()
        pygame.display.set_caption("QWOP")
        screen = pygame.display.set_mode(size)
        
        #prepare background surface
        background_surface = animator.drawBackground(width,height)
        screen.blit(background_surface, (0, 0))
        pygame.display.update()
        
        
        
        #loop 
        time_steps = len(state_log)
        for this_step in range(time_steps): #this loop becomes while not dead for game
            print(this_step)
            
            #get current state (from state or using physics in game)
            
            
            #Prep surface
            #frame_surface = animator.drawState(background_surface,self, state_log(i), action_log(i), episode_number)
            #Draw new body
        
            #save image
            pygame.image.save(background_surface,"test.png")
            
        pygame.quit()
        
        #read images and write video, delete images
        