#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:50:29 2019

@author: StephaneMagnan
"""
import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'
from os.path import isfile, join
import cv2
import numpy as np
import pygame
from settings import Settings
# Importing the environment
Environment = __import__('environment_' + Settings.ENVIRONMENT).Environment

def runCode():
    #runfile('/Users/StephaneMagnan/Documents/GitHub/QWOP/animator.py', wdir='/Users/StephaneMagnan/Documents/GitHub/QWOP')
    import animator
    animator.drawState(0,800,500)
    
def containerTest(var):
    # intialize window -
    width = 800
    height = 500
    pygame.init()
    pygame.display.set_caption("QWOP")
    # Set the height and width of the screen
    size = [width, height]
    screen = pygame.display.set_mode(size)
    
    background_surface = drawBackground(width,height)
    screen.blit(background_surface, (0, 0))
    

    screen = pygame.display.set_mode(size)
    
    
    pygame.display.update()
    
#    # Save every frame
#    file_num = 0
#    this_surface = 0
#    filename = "Snaps/%04d.png" % file_num
#    pygame.image.save(this_surface, filename)
    return 0

def returnPointCoords(x,y,theta,l,gamma,x_c,x_0,y_0,hum_scale):
    #gamma is "above", eta is "below" CG
    
    pointCoords = np.array([[x-gamma*np.sin(theta),y+gamma*np.cos(theta)],[x,y],[x+(l-gamma)*np.sin(theta),y-(l-gamma)*np.cos(theta)]])      
    
    pointCoords[:,0]=(pointCoords[:,0]-x_c)*hum_scale+x_0
    pointCoords[:,1]=y_0-(pointCoords[:,1])*hum_scale
    
    return pointCoords
    
   
    
def parseAction(action):
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
    else: #action == 0:
        pressed_q = False
        pressed_w = False
        pressed_o = False
        pressed_p = False
    
    return pressed_q,pressed_w,pressed_o,pressed_p


def packAction(pressed_q,pressed_w,pressed_o,pressed_p):                 
    #0: No buttons pressed; 1: Q only; 2: QO; 3: QP; 4: W only; 5: WO; 6: WP; 7: O only; 8: P only
    action = 0
    
    #opposite buttons cannot both be true
    if pressed_q and pressed_w:
        pressed_q = False
        pressed_w = False
    if pressed_o and pressed_p:
        pressed_o = False
        pressed_p = False
        
    #determine aciton ID    
    if       pressed_q and not pressed_w and not pressed_o and not pressed_p:
        action = 1
    elif     pressed_q and not pressed_w and     pressed_o and not pressed_p:
        action = 2 
    elif     pressed_q and not pressed_w and not pressed_o and     pressed_p:
        action = 3 
    elif not pressed_q and     pressed_w and not pressed_o and not pressed_p:
        action = 4 
    elif not pressed_q and     pressed_w and     pressed_o and not pressed_p:
        action = 5 
    elif not pressed_q and     pressed_w and not pressed_o and     pressed_p:
        action = 6 
    elif not pressed_q and not pressed_w and     pressed_o and not pressed_p:
        action = 7 
    elif not pressed_q and not pressed_w and not pressed_o and     pressed_p:
        action = 8 
    else:
        action = 0

    return action     


def drawBackground(width,height):
    
    #define colors
    DIRT = (58,28,18)
    TRACK = (208,69,40)
    LINE = (255,255,255)
    GRASS = (100,150,50)
    SKY = (69,99,119)
    SKY2 = (0,102,162)
    TEXT_RELEASED = (55, 55, 55)
    RELEASED = (200,200,200)
    
    #define relative positions of background
    y_sky = np.rint(0.5*height)
    y_sky2 = np.rint(0.075*height)
    y_grass = np.rint(0.175*height)
    y_track = np.rint(0.2*height)
    y_dirt = height-np.min((y_sky+y_sky2+y_grass+y_track,height))
    
    #define relative position of track lines
    y_l1 = y_sky+y_sky2+y_grass 
    y_l4 = y_sky+y_sky2+y_grass+y_track
    y_l2 = y_l1+np.rint((y_l4-y_l1)/6)
    y_l3 = y_l1+np.rint((y_l4-y_l1)/2)
    #define thickness of track lines
    t_l1 = 1
    t_l2 = 2
    t_l3 = 3
    t_l4 = 5
    
    #define the location of buttons
    dx_btn = np.rint(height/12)
    dy_btn = np.rint(height/12)

    x_btn1 = np.rint(width/15)+20
    x_btn2 = x_btn1+dx_btn+20
    x_btn3 = width-x_btn2-dx_btn
    x_btn4 =  width-x_btn1-dx_btn
    y_btn = np.rint(1*height/10)
    
    background_surface = pygame.Surface((width, height))
    screen = background_surface
    
    # --- Drawing
    # Set the screen background (clears)
    screen.fill(SKY)
 
    # Draw the background
    #pygame.draw.rect(screen, SKY, [0,0,width,y_sky])
    pygame.draw.rect(screen, SKY2, [0,y_sky,width,y_sky2])
    pygame.draw.rect(screen, GRASS, [0,y_sky+y_sky2,width,y_grass])
    pygame.draw.rect(screen, TRACK, [0,y_sky+y_sky2+y_grass,width,y_track])
    pygame.draw.rect(screen, DIRT, [0,y_sky+y_sky2+y_grass+y_track,width,y_dirt])
    
    # Draw track lines
    pygame.draw.line(screen, LINE, [0, y_l1], [width, y_l1], t_l1)
    pygame.draw.line(screen, LINE, [0, y_l2], [width, y_l2], t_l2)
    pygame.draw.line(screen, LINE, [0, y_l3], [width, y_l3], t_l3)
    pygame.draw.line(screen, LINE, [0, y_l4], [width, y_l4], t_l4)
    
    
    # Select the font to use, size, bold, italics
    font = pygame.font.SysFont('courier', 25, True, False)
 
    # Draw default buttons (released)
    pygame.draw.rect(screen, RELEASED, [x_btn1,y_btn,dx_btn,dy_btn])
    text_q = font.render("Q", True, TEXT_RELEASED)
    text_q_w = text_q.get_rect().width
    text_q_h = text_q.get_rect().height
    screen.blit(text_q, [x_btn1+np.rint(dx_btn/2-text_q_w/2), y_btn+np.rint(dy_btn/2-text_q_h/2)])
    
    pygame.draw.rect(screen, RELEASED, [x_btn2,y_btn,dx_btn,dy_btn])
    text_w = font.render("W", True, TEXT_RELEASED)
    text_w_w = text_w.get_rect().width
    text_w_h = text_w.get_rect().height
    screen.blit(text_w, [x_btn2+np.rint(dx_btn/2-text_w_w/2), y_btn+np.rint(dy_btn/2-text_w_h/2)])
    
    pygame.draw.rect(screen, RELEASED, [x_btn3,y_btn,dx_btn,dy_btn])
    text_o = font.render("O", True, TEXT_RELEASED)
    text_o_w = text_o.get_rect().width
    text_o_h = text_o.get_rect().height
    screen.blit(text_o, [x_btn3+np.rint(dx_btn/2-text_o_w/2), y_btn+np.rint(dy_btn/2-text_o_h/2)])
    
    pygame.draw.rect(screen, RELEASED, [x_btn4,y_btn,dx_btn,dy_btn])   
    text_p = font.render("P", True, TEXT_RELEASED)
    text_p_w = text_p.get_rect().width
    text_p_h = text_p.get_rect().height
    screen.blit(text_p, [x_btn4+np.rint(dx_btn/2-text_p_w/2), y_btn+np.rint(dy_btn/2-text_p_h/2)])
   
    return background_surface

def drawDistLine(width,hum_scale,x):
    #this function returns a set of points for drawing lines every 5 meters
    line_points = []

    #determine number of meters above x_c, and number below x_c
    half_width = width/2    
    
    #determine distance above and below to next multiple of 5
    offset_x = np.mod(x,5)
    low_x = -offset_x
    high_x = +(5-offset_x)
    
    #convert to pixels offset from centre
    low_px = half_width+low_x*hum_scale
    high_px = half_width+high_x*hum_scale
    
    #save the points to file include px and x 
    line_points.append([low_px,x+low_x]) 
    line_points.append([high_px,x+high_x]) 
    
    return line_points

        
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #remove non png files
    file_counter = 0
    while file_counter < len(files):

        if files[file_counter][-4:] != ".png":
            #print("Removing " + files[file_counter])
            files.remove(files[file_counter])
        else:
            file_counter += 1 # only increment counter if a file was not removed
    
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[-9:-4]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
        os.remove(filename)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()  
    

def drawState(play_game, filename="", state_log=None, action_log=None, episode_number=1):

    
    #define game parameters
    print_out = False

    #save_game = False
    #play_game = True
    begin = False
    game_over = False
    quit_game = False
    
    if filename == "":
        save_game = False
    else:
        save_game = True
        if play_game:
            save_path = "frames/"+filename +"/"
        else:
            save_path = Settings.MODEL_SAVE_DIRECTORY + filename + "/videos/"  
    
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    
    max_frame = 2000    
    this_time = 0
    
    # generation                best x (gen)
    # time      
    best_x = 0
    best_trial = 0
    
    
    # intialize window 
    pygame.init()
    pygame.display.set_caption("QWOP")
    
    # Set the height and width of the screen
    width = 800
    height = 500
    size = [width, height]
    hum_scale = 150 #pixel/m
    screen = pygame.display.set_mode(size)
    background = drawBackground(width,height)
    
    #initialize the environment
    env = Environment() #create an instance of the environment
    state = env.reset() #resets the environement and puts the initial condiditons in state
    
    
    # Define some colors
    #BLACK = (0, 0, 0)
    #WHITE = (255, 255, 255)
    #GREEN = (0, 255, 0)
    #RED = (255, 0, 0)
    
    TEXT = (255, 255, 255)
    TEXT_PRESSED = (200, 200, 200)
    PRESSED = (50, 50, 50)   
    TEXT_GAMEOVER = (255, 25, 25)   
    TEXT_BEGIN = (200, 200, 200)
    
    LINE = (255,255,255)
    BODY = (25, 25, 25)
    COG = (200,200,200)
    #GAMEOVER = (255,50,50)
    
    # Define some fonts to use, size, bold, italics
    font_subtitles = pygame.font.SysFont('courier', 18, True, False)
    font_begin = pygame.font.SysFont('courier', 16, True, False)
    font_distance = pygame.font.SysFont('courier', 30, True, False)
    font_ticks = pygame.font.SysFont('courier', 12, True, False)   
    font_gameover = pygame.font.SysFont('courier', 48, True, False)
    font_qwop = pygame.font.SysFont('courier', 25, True, False)
    
    text_q = font_qwop.render("Q", True, TEXT_PRESSED)
    text_q_w = text_q.get_rect().width
    text_qwop_h = text_q.get_rect().height
    
    text_w = font_qwop.render("W", True, TEXT_PRESSED)
    text_w_w = text_q.get_rect().width
    #text_w_h = text_q.get_rect().height 
    
    text_o = font_qwop.render("O", True, TEXT_PRESSED)
    text_o_w = text_q.get_rect().width
    #text_o_h = text_q.get_rect().height
    
    text_p = font_qwop.render("P", True, TEXT_PRESSED)
    text_p_w = text_q.get_rect().width
    #text_p_h = text_q.get_rect().height
            


    #define relative positions of text and buttons    
    dx_btn = np.rint(height/12)
    dy_btn = np.rint(height/12)
    
    x_btn1 = np.rint(width/15)+20
    x_btn2 = x_btn1+dx_btn+20
    x_btn3 = width-x_btn2-dx_btn
    x_btn4 =  width-x_btn1-dx_btn
    y_btn = np.rint(1*height/10)
    
    
    x_btnq = x_btn1+np.rint(dx_btn/2-text_q_w/2)
    x_btnw = x_btn2+np.rint(dx_btn/2-text_w_w/2)
    x_btno = x_btn3+np.rint(dx_btn/2-text_o_w/2)
    x_btnp = x_btn4+np.rint(dx_btn/2-text_p_w/2)
    y_btnqwop =  y_btn+np.rint(dy_btn/2-text_qwop_h/2)
    
    #define position of painted lines
    y_l1 = 0.75*height
    y_l4 = 0.95*height
    y_lx = 0.97*height
    
    #define positions of body
    x_0 = np.rint(width/2)
    y_0 = np.rint(height*9/10)
    
    


     
    # Loop until the user clicks the close button.
    running = True
    pressed_q = False
    pressed_w = False
    pressed_o = False
    pressed_p = False
    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks()
    
#    myimage = pygame.image.load("background_im.png")
#    imagerect = myimage.get_rect()
    
    
    #define the body (segment coordinates)
    segment_count = 9
    segment_points = np.zeros((segment_count,3,2))
    #print(state)
#    x,y,theta =       state[0:3]
#    x1r,y1r,theta1r = state[3:6]
#    x2r,y2r,theta2r = state[6:9]
#    x3r,y3r,theta3r = state[9:12]
#    x4r,y4r,theta4r = state[12:15]
#    xfr,yfr =         state[15:17]
#    x1l,y1l,theta1l = state[17:20]
#    x2l,y2l,theta2l = state[20:23]
#    x3l,y3l,theta3l = state[23:26]
#    x4l,y4l,theta4l = state[26:29]
#    xfl,yfl =         state[29:31]
    
    #unpack state
    x,y,theta = state[0:3]
    x1r,y1r,theta1r = state[3:6]
    x2r,y2r,theta2r = state[6:9]
    x3r,y3r,theta3r = state[9:12]
    x4r,y4r,theta4r = state[12:15]
    #xfr,yfr = next_state[15:17]
    x1l,y1l,theta1l = state[15:18]
    x2l,y2l,theta2l = state[18:21]
    x3l,y3l,theta3l = state[21:24]
    x4l,y4l,theta4l = state[24:27]
    #xfl,yfl = next_state[29:31]
    
    
    # DRAW INITIAL STATE 
    l=env.SEGMENT_LENGTH[0]
    l1=env.SEGMENT_LENGTH[1]
    l2=env.SEGMENT_LENGTH[2]
    l3=env.SEGMENT_LENGTH[3]
    l4=env.SEGMENT_LENGTH[4]
    gamma=env.SEGMENT_GAMMA_LENGTH[0]
    gamma1=env.SEGMENT_GAMMA_LENGTH[1]
    gamma2=env.SEGMENT_GAMMA_LENGTH[2]
    gamma3=env.SEGMENT_GAMMA_LENGTH[3]
    gamma4=env.SEGMENT_GAMMA_LENGTH[4]
    #eta=env.SEGMENT_ETA_LENGTH[0]
    #eta1=env.SEGMENT_ETA_LENGTH[1]
    #eta2=env.SEGMENT_ETA_LENGTH[2]
    #eta3=env.SEGMENT_ETA_LENGTH[3]
    #eta4=env.SEGMENT_ETA_LENGTH[4]
    
        
    #prepare screen
    screen.blit(background,(0,0))
    #draw stickman in initial position
    #Get point coordinates for each segment
    segment_points[0,:,:] = returnPointCoords(x,  y,  theta,  l, gamma, x,x_0,y_0,hum_scale)
    segment_points[1,:,:] = returnPointCoords(x1r,y1r,theta1r,l1,gamma1,x,x_0,y_0,hum_scale)
    segment_points[2,:,:] = returnPointCoords(x2r,y2r,theta2r,l2,gamma2,x,x_0,y_0,hum_scale)
    segment_points[3,:,:] = returnPointCoords(x3r,y3r,theta3r,l3,gamma3,x,x_0,y_0,hum_scale)
    segment_points[4,:,:] = returnPointCoords(x4r,y4r,theta4r,l4,gamma4,x,x_0,y_0,hum_scale)
    segment_points[5,:,:] = returnPointCoords(x1l,y1l,theta1l,l1,gamma1,x,x_0,y_0,hum_scale)
    segment_points[6,:,:] = returnPointCoords(x2l,y2l,theta2l,l2,gamma2,x,x_0,y_0,hum_scale)
    segment_points[7,:,:] = returnPointCoords(x3l,y3l,theta3l,l3,gamma3,x,x_0,y_0,hum_scale)
    segment_points[8,:,:] = returnPointCoords(x4l,y4l,theta4l,l4,gamma4,x,x_0,y_0,hum_scale)
    
    
    headCoords = np.array([[x-gamma*np.sin(theta),y+gamma*np.cos(theta)],[x-(gamma+env.NECK_LENGTH)*np.sin(theta),y+(gamma+env.NECK_LENGTH)*np.cos(theta)]])      
    
    #determine point coordinates for head and neck
    headCoords[:,0]=(headCoords[:,0]-x)*hum_scale+x_0
    headCoords[:,1]=y_0-(headCoords[:,1])*hum_scale


    
    
    
    #Draw body
    for segment_id in range(segment_count):
        pygame.draw.line(screen, BODY, segment_points[segment_id,0,:], segment_points[segment_id,2,:], 10)
        pygame.draw.ellipse(screen, COG, [segment_points[segment_id,1,0]-5,segment_points[segment_id,1,1]-5,10,10], 2)
        pygame.draw.ellipse(screen, BODY, [segment_points[segment_id,0,0]-6,segment_points[segment_id,0,1]-6,12,12], 0)
        pygame.draw.ellipse(screen, BODY, [segment_points[segment_id,2,0]-6,segment_points[segment_id,2,1]-6,12,12], 0)
    
    #draw head and neck
    pygame.draw.line(screen, BODY, headCoords[0,:], headCoords[1,:], 10)
    pygame.draw.ellipse(screen, BODY, [headCoords[1,0]-np.round(env.HEAD_DIAM/2*hum_scale),headCoords[1,1]-np.round(env.HEAD_DIAM/2*hum_scale),np.round(env.HEAD_DIAM*hum_scale),np.round(env.HEAD_DIAM*hum_scale)])

       
    
    if play_game:
        text_begin = font_begin.render("PRESS [SPACE] TO PLAY", True, TEXT_BEGIN)
        screen.blit(text_begin, [10,10])
        
    #force screen to display all the latest    
    pygame.display.flip()
    
    
    frame_num = 0
    # -------- Main Program Loop -----------
    while running:
        if play_game:
            ############################################            
            ###### DETERMINE USER  INPUTS TO GAME ######            
            ############################################   
            
            #pygame.event.wait()
            # --- Event Processing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    #print("User pressed a key.")
                    keys = pygame.key.get_pressed()
        
                    pressed_q = keys[pygame.K_q]
                    pressed_w = keys[pygame.K_w]
                    pressed_o = keys[pygame.K_o]
                    pressed_p = keys[pygame.K_p]
                    
                    if print_out:
                        if pressed_q:
                            print("Q")
                        if pressed_w:
                            print("W")
                        if pressed_o:
                            print("O")
                        if pressed_p:
                            print("P")
                            
                    if keys[pygame.K_SPACE]:
                        if not begin:
                            begin = True
                        if game_over:
                            
                            game_over = False
                            #check high-score
                            if x >=best_x:
                                best_x = x
                                best_trial = episode_number
                                
                            
                            #update time and texts
                            episode_number +=1
                            this_time=0
                            #reset environment
                            env.reset()
                            
                    if keys[pygame.K_ESCAPE]:
                        if game_over:
                            quit_game = True
                        
                elif event.type == pygame.KEYUP:
                    #print("User let go of a key.")
                    keys = pygame.key.get_pressed()
                    
                    if print_out:
                        if not keys[pygame.K_q]:
                            if pressed_q:
                                print("!Q")
                        if not keys[pygame.K_w]:
                            if pressed_w:
                                print("!W")
                        if not keys[pygame.K_o]:
                            if pressed_o:
                                print("!O")
                        if not keys[pygame.K_p]:
                            if pressed_p:
                                print("!P")  
                        
                    pressed_q = keys[pygame.K_q]
                    pressed_w = keys[pygame.K_w]
                    pressed_o = keys[pygame.K_o]
                    pressed_p = keys[pygame.K_p]        
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if print_out:
                         print("User pressed a mouse button")
                elif event.type == pygame.MOUSEMOTION:
                    if print_out:
                         print("User moved the mouse")            
                         po = pygame.mouse.get_pos()
                         print(po)
                else:
                    if print_out:
                         print("Other")
                     
        if begin or not play_game:   
            ############################################            
            ####### GAME LOGIC - DETERMINE STATE #######            
            ############################################ 
            
            # --- Logic
            if play_game:
            
                #pack button presses into integer tag
                this_action = packAction(pressed_q,pressed_w,pressed_o,pressed_p)
                
                #Step the dynamics forward one timestep
                next_state,reward,game_over = env.step(this_action)
                #done = False
            
            else: #not play game
                #look at the state vector history and figure out what to do
                
                

                
                if frame_num == np.size(state_log, axis=0)-1:                
                    quit_game = True
                else:                    
                    #dissect state
                    next_state = state_log[frame_num]
                    #dissect actions
                    pressed_q,pressed_w,pressed_o,pressed_p = parseAction(action_log[frame_num])
                
                    
            
            
            #DONE GETS A PROPMOTION, REMOVE DONE CHECK FROM THIS CODE - USE GAME_OVER
            
            #unpack state
            x,y,theta = next_state[0:3]
            x1r,y1r,theta1r = next_state[3:6]
            x2r,y2r,theta2r = next_state[6:9]
            x3r,y3r,theta3r = next_state[9:12]
            x4r,y4r,theta4r = next_state[12:15]
            #xfr,yfr = next_state[15:17]
            x1l,y1l,theta1l = next_state[15:18]
            x2l,y2l,theta2l = next_state[18:21]
            x3l,y3l,theta3l = next_state[21:24]
            x4l,y4l,theta4l = next_state[24:27]
            #xfl,yfl = next_state[29:31]

            #Get point coordinates for each segment
            segment_points[0,:,:] = returnPointCoords(x,  y,  theta,  l, gamma, x,x_0,y_0,hum_scale)
            segment_points[1,:,:] = returnPointCoords(x1r,y1r,theta1r,l1,gamma1,x,x_0,y_0,hum_scale)
            segment_points[2,:,:] = returnPointCoords(x2r,y2r,theta2r,l2,gamma2,x,x_0,y_0,hum_scale)
            segment_points[3,:,:] = returnPointCoords(x3r,y3r,theta3r,l3,gamma3,x,x_0,y_0,hum_scale)
            segment_points[4,:,:] = returnPointCoords(x4r,y4r,theta4r,l4,gamma4,x,x_0,y_0,hum_scale)
            segment_points[5,:,:] = returnPointCoords(x1l,y1l,theta1l,l1,gamma1,x,x_0,y_0,hum_scale)
            segment_points[6,:,:] = returnPointCoords(x2l,y2l,theta2l,l2,gamma2,x,x_0,y_0,hum_scale)
            segment_points[7,:,:] = returnPointCoords(x3l,y3l,theta3l,l3,gamma3,x,x_0,y_0,hum_scale)
            segment_points[8,:,:] = returnPointCoords(x4l,y4l,theta4l,l4,gamma4,x,x_0,y_0,hum_scale)

            headCoords = np.array([[x-gamma*np.sin(theta),y+gamma*np.cos(theta)],[x-(gamma+env.NECK_LENGTH)*np.sin(theta),y+(gamma+env.NECK_LENGTH)*np.cos(theta)]])      
    
            #determine point coordinates for head and neck
            headCoords[:,0]=(headCoords[:,0]-x)*hum_scale+x_0
            headCoords[:,1]=y_0-(headCoords[:,1])*hum_scale



            # Determine location of painted lines based on X distance 
            line_points = drawDistLine(width,hum_scale,x)
            
            
        
            ############################################            
            ####### DRAW THE STATE TO THE SCREEN #######            
            ############################################            
            #INDEPENDENT OF CASE
            
            # --- Drawing
            # Set the screen background (clears)
            #screen.fill(BLACK)
            screen.blit(background,(0,0))   
            
            
            #Draw text
            
            # write time and record subtitles
            this_time = this_time#(pygame.time.get_ticks() - start_time)/1000
            text_time = font_subtitles.render("%3.2f s" %this_time, True, TEXT)
            text_t_w = text_time.get_rect().width
            screen.blit(text_time, [x_btn4+dx_btn-text_t_w,y_btn-dy_btn])
            
            if play_game:
                text_record = font_subtitles.render("%3.2f m (%i)" % (best_x, best_trial), True, TEXT)
                screen.blit(text_record, [x_btn1,y_btn-dy_btn])
    
            # write current score title
            text_current = font_distance.render("%3.2f m (%i)" %(x,episode_number), True, TEXT)
            text_c_w = text_current.get_rect().width
            screen.blit(text_current, [np.rint(width/2-text_c_w/2),y_btn])
    
            #Draw buttons (coloured only, u coloured is background)
            if pressed_q:
                pygame.draw.rect(screen, PRESSED, [x_btn1,y_btn,dx_btn,dy_btn])
                text_q = font_qwop.render("Q", True, TEXT_PRESSED)
                screen.blit(text_q, [x_btnq, y_btnqwop])
                if print_out:
                    print("Q pressed")
            if pressed_w:
                pygame.draw.rect(screen, PRESSED, [x_btn2,y_btn,dx_btn,dy_btn])
                text_w = font_qwop.render("W", True, TEXT_PRESSED)
                screen.blit(text_w, [x_btnw, y_btnqwop])
                if print_out:
                    print("W pressed")
            if pressed_o:
                pygame.draw.rect(screen, PRESSED, [x_btn3,y_btn,dx_btn,dy_btn])
                text_o = font_qwop.render("O", True, TEXT_PRESSED)
                screen.blit(text_o, [x_btno, y_btnqwop])
                if print_out:
                    print("O pressed")
            if pressed_p:
                pygame.draw.rect(screen, PRESSED, [x_btn4,y_btn,dx_btn,dy_btn]) 
                text_p = font_qwop.render("P", True, TEXT_PRESSED)
                screen.blit(text_p, [x_btnp, y_btnqwop])
                if print_out:
                    print("P pressed")
                
                
            #Draw distance ticks
            for this_line in range(len(line_points)):
                pygame.draw.line(screen, LINE, [line_points[this_line][0],y_l1], [line_points[this_line][0],y_l4],3)
                text_tick = font_ticks.render(str(int(line_points[this_line][1])), True, TEXT)
                text_tick_w = text_tick.get_rect().width
                screen.blit(text_tick, [line_points[this_line][0]-np.rint(text_tick_w/2), y_lx])
            
            
            #Draw body
            #print(frame)
            for segment_id in range(segment_count):
                pygame.draw.line(screen, BODY, segment_points[segment_id,0,:], segment_points[segment_id,2,:], 10)
                pygame.draw.ellipse(screen, COG, [segment_points[segment_id,1,0]-5,segment_points[segment_id,1,1]-5,10,10], 2)
                pygame.draw.ellipse(screen, BODY, [segment_points[segment_id,0,0]-6,segment_points[segment_id,0,1]-6,12,12], 0)
                pygame.draw.ellipse(screen, BODY, [segment_points[segment_id,2,0]-6,segment_points[segment_id,2,1]-6,12,12], 0)
                #print(" ", segment_points[segment_id,0,:],segment_points[segment_id,2,:])
   
                #draw head and neck
                pygame.draw.line(screen, BODY, headCoords[0,:], headCoords[1,:], 10)
                pygame.draw.ellipse(screen, BODY, [headCoords[1,0]-np.round(env.HEAD_DIAM/2*hum_scale),headCoords[1,1]-np.round(env.HEAD_DIAM/2*hum_scale),np.round(env.HEAD_DIAM*hum_scale),np.round(env.HEAD_DIAM*hum_scale)])



         
            #check if node out-of-bounds. If so, call it quits 
            if np.any(segment_points[:,0,1]>height):
                text_GO = font_gameover.render("GAME OVER!", True, TEXT_GAMEOVER)
                
                text_GO_w = text_GO.get_rect().width
                text_GO_h = text_GO.get_rect().height
                screen.blit(text_GO, [np.rint(width/2-text_GO_w/2), np.rint(height/2-text_GO_h/2)])
                
                game_over = True
                   
                
        
            # --- Wrap-up
            # Limit to 60 frames per second
            if play_game:
                clock.tick(20)
         
            # Go ahead and update the screen with what we've drawn.
            pygame.display.flip() # more efficient: update(rectangle_list) https://www.pygame.org/docs/ref/display.html
        
             
            
            #save frame to file
            if save_game:
                im_name = save_path+ "frame_%05i.png" %frame_num
                #print(im_name)
                pygame.image.save(screen,im_name)          
            
            #Check if the game is over (dynamics are complete)
            if quit_game:
                break
            
            #check if too many iterations have occured
            frame_num+=1
            if frame_num >=max_frame:
                break
        
        
            this_time += env.TIMESTEP
    #save to video
    pathIn= save_path
    pathOut = save_path + '/episode_' + str(episode_number) + '.avi'
    fps = 1/env.TIMESTEP
    convert_frames_to_video(pathIn, pathOut, fps)
    
    
    # Close everything down
    pygame.quit()
    
    #print("quit")
    
    

 