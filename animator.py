#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:50:29 2019

@author: StephaneMagnan
"""

import numpy as np
import pygame
from environment_qwop import Environment

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

def returnPointCoords(x,y,theta_cum,gamma,eta,x_c,x_0,y_0,hum_scale):
    #gamma is "above", eta is "below" CG
    
    pointCoords = np.array([[x-gamma*np.sin(theta_cum),y+gamma*np.cos(theta_cum)],[x,y],[x+eta*np.sin(theta_cum),y-eta*np.cos(theta_cum)]])      
    
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

        


def drawState(state,width,height):
    # intialize window 
    pygame.init()
    pygame.display.set_caption("QWOP")
    
    # Set the height and width of the screen
    size = [width, height]
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
    
    LINE = (255,255,255)
    BODY = (25, 25, 25)
    COG = (200,200,200)
    #GAMEOVER = (255,50,50)
    
    # Define some fonts to use, size, bold, italics
    font_subtitles = pygame.font.SysFont('courier', 18, True, False)
    font_distance = pygame.font.SysFont('courier', 30, True, False)
    font_ticks = pygame.font.SysFont('courier', 12, True, False)
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
            

        
    # generation                best x (gen)
    # time                      x
    n_trial = 1
    best_x = 20
    best_trial = 10
    
    #scale
    hum_scale = 100 #pixel/m
    width = 800
    height = 500
    
    
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
    
    #define the body
    segment_count = 3
    segment_points = np.zeros((segment_count,3,2))


    l = 1#self.body_length
    l1 = 1#self.leg1_length
    l2 = 1#self.leg2_length  
    
    a = 0.5
    a1 = 0.55
    a2 = 0.55
    
    x= 0.00000000e+00
    y= 2.00000000e+00
    theta= 0 
    x1=2.50000000e-01
    y1=1.06698730e+00
    theta1=30*np.pi/180
    x2=-2.50000000e-01 
    y2=1.06698730e+00
    theta2=-30*np.pi/180

     
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
    
    print_out = False
    
    # -------- Main Program Loop -----------
    while running:
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
                     

        # --- Logic


        # Debugging, change angle of bodies
#        if pressed_q:
#            theta1 = theta1 + 1*np.pi/180
#            theta2 = theta2 - 1*np.pi/180
#        if pressed_w:
#            theta1 = theta1 - 1*np.pi/180
#            theta2 = theta2 + 1*np.pi/180
#        if pressed_o:
#            theta = theta + 1*np.pi/180
#        if pressed_p:
#            theta = theta - 1*np.pi/180
        
        #pack button presses into integer tag
        this_action = packAction(pressed_q,pressed_w,pressed_o,pressed_p)
        

        #Step the dynamics forward one timestep
        next_state,reward,done = env.step(this_action)
        #done = False
        
        #print(next_state)
        
        
        x = next_state[0]
        y = next_state[1]
        theta = next_state[2]
        x1 = next_state[3]
        y1 = next_state[4]
        theta1 = next_state[5]
        x2 = next_state[6]
        y2 = next_state[7]
        theta2 = next_state[8]
        
        #Get point coordinates for each segment
        segment_points[0,:,:] = returnPointCoords(x,y,theta,a*l,(1-a)*l,x,x_0,y_0,hum_scale)
        segment_points[1,:,:] = returnPointCoords(x1,y1,theta+theta1,a1*l1,(1-a1)*l1,x,x_0,y_0,hum_scale)
        segment_points[2,:,:] = returnPointCoords(x2,y2,theta+theta2,a2*l2,(1-a2)*l2,x,x_0,y_0,hum_scale)
        
        # Determine location of painted lines based on X distance 
        line_points = drawDistLine(width,hum_scale,x)
        
            
        # --- Drawing
        # Set the screen background (clears)
        #screen.fill(BLACK)
        screen.blit(background,(0,0))   
        
        
        #Draw text
        
        # write time and record subtitles
        this_time = (pygame.time.get_ticks() - start_time)/1000
        text_time = font_subtitles.render("%3.2fs" %this_time, True, TEXT)
        text_t_w = text_time.get_rect().width
        screen.blit(text_time, [x_btn4+dx_btn-text_t_w,y_btn-dy_btn])
        
        text_record = font_subtitles.render("%3.2fm (%i)" % (best_x, best_trial), True, TEXT)
        screen.blit(text_record, [x_btn1,y_btn-dy_btn])

        # write current score title
        text_current = font_distance.render("%3.2fm (%i)" %(x,n_trial), True, TEXT)
        text_c_w = text_current.get_rect().width
        screen.blit(text_current, [np.rint(width/2-text_c_w/2),y_btn])

        #Draw buttons (coloured only, u coloured is background)
        if pressed_q:
            pygame.draw.rect(screen, PRESSED, [x_btn1,y_btn,dx_btn,dy_btn])
            text_q = font_qwop.render("Q", True, TEXT_PRESSED)
            screen.blit(text_q, [x_btnq, y_btnqwop])
            if print_out:
                x+=0.01
                x1+=0.01
                x2+=0.01
        if pressed_w:
            pygame.draw.rect(screen, PRESSED, [x_btn2,y_btn,dx_btn,dy_btn])
            text_w = font_qwop.render("W", True, TEXT_PRESSED)
            screen.blit(text_w, [x_btnw, y_btnqwop])
            if print_out:
                x-=0.01
                x1-=0.01
                x2-=0.01
        if pressed_o:
            pygame.draw.rect(screen, PRESSED, [x_btn3,y_btn,dx_btn,dy_btn])
            text_o = font_qwop.render("O", True, TEXT_PRESSED)
            screen.blit(text_o, [x_btno, y_btnqwop])
            if print_out:
                x+=0.05
                x1+=0.05
                x2+=0.05
        if pressed_p:
            pygame.draw.rect(screen, PRESSED, [x_btn4,y_btn,dx_btn,dy_btn]) 
            text_p = font_qwop.render("P", True, TEXT_PRESSED)
            screen.blit(text_p, [x_btnp, y_btnqwop])
            if print_out:
                x-=0.05
                x1-=0.05
                x2-=0.05
            
            
        #Draw distance ticks
        for this_line in range(len(line_points)):
            pygame.draw.line(screen, LINE, [line_points[this_line][0],y_l1], [line_points[this_line][0],y_l4],3)
            text_tick = font_ticks.render(str(int(line_points[this_line][1])), True, TEXT)
            text_tick_w = text_tick.get_rect().width
            screen.blit(text_tick, [line_points[this_line][0]-np.rint(text_tick_w/2), y_lx])
        
        
        #Draw body
        for segment_id in range(segment_count):
            pygame.draw.line(screen, BODY, segment_points[segment_id,0,:], segment_points[segment_id,2,:], 10)
            pygame.draw.ellipse(screen, COG, [segment_points[segment_id,1,0]-5,segment_points[segment_id,1,1]-5,10,10], 2)
            pygame.draw.ellipse(screen, BODY, [segment_points[segment_id,0,0]-6,segment_points[segment_id,0,1]-6,12,12], 0)
            pygame.draw.ellipse(screen, BODY, [segment_points[segment_id,2,0]-6,segment_points[segment_id,2,1]-6,12,12], 0)
            
            
    
        # --- Wrap-up
        # Limit to 60 frames per second
        clock.tick(60)
     
        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip() # more efficient: update(rectangle_list) https://www.pygame.org/docs/ref/display.html
        
        #Check if the dynamics are complete
        if done:
            break
        
    # Close everything down
    pygame.quit()
    
    print("quit")
    
    
    
 