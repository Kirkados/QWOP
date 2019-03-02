#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:50:29 2019

@author: StephaneMagnan
"""

import numpy as np
import pygame
 
def something(var):
    # intialize window 
    width = 800
    height = 500
    pygame.init()
    pygame.display.set_caption("QWOP")
    # Set the height and width of the screen
    size = [width, height]
    screen = pygame.display.set_mode(size)
    
    background_surface = animator.drawBackground(width,height)
    screen.blit(background_surface, (0, 0))
    

    screen = pygame.display.set_mode(size)
    
    
    pygame.display.update()
    
    # Save every frame
    file_num = 0
    this_surface = 0
    filename = "Snaps/%04d.png" % file_num
    pygame.image.save(this_surface, filename)
    return 0

def returnPointCoords(x,y,theta_cum,gamma,eta):
    
    pointCoords = np.array([[x-(1-a_bod)*l_bod*np.sin(theta_cum),y+(1-a_bod)*l_bod*np.cos(theta_cum)],[x,y],[x+(a_bod)*l_bod*np.sin(theta_cum),y-(a_bod)*l_bod*np.cos(theta_cum)]])      
        
    
    return pointCoords
    

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
    print(y_l3,y_l4)
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


def drawState(state):
    # Define dimensions for background
    
    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    
    TEXT = (255, 255, 255)
    TEXT_PRESSED = (200, 200, 200)
    PRESSED = (50, 50, 50)
    
    LINE = (255,255,255)
    BODY = (25, 25, 25)
    COG = (200,200,200)
    GAMEOVER = (255,50,50)
    
    
    
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
    
    #define positions of body
    x_0 = np.rint(width/2)
    y_0 = np.rint(height*9/10)
    
    x_bod = 10
    y_bod = 1.4
    th_bod = 0
    x_bod1 = 15
    y_bod1 = 0.4
    th_leg1 = 30*np.pi/180
    x_bod2 = 5
    y_bod2 = 0.4
    th_leg2 = -30*np.pi/180
    

    # intialize window 
    pygame.init()
    pygame.display.set_caption("QWOP")
    
    # Set the height and width of the screen
    size = [width, height]
    screen = pygame.display.set_mode(size)
    background = drawBackground(width,height)
     
    # Loop until the user clicks the close button.
    running = True
    pressed_q = False
    pressed_w = False
    pressed_o = False
    pressed_p = False
    # Used to manage how fast the screen updates
    #clock = pygame.time.Clock()
    
    
#    myimage = pygame.image.load("background_im.png")
#    imagerect = myimage.get_rect()
    
    
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
    
                pressed_q = False
                pressed_w = False
                pressed_o = False
                pressed_p = False
                
                if keys[pygame.K_q]:
                    pressed_q = True
                    print("Q")
                if keys[pygame.K_w]:
                    pressed_w = True
                    print("W")
                if keys[pygame.K_o]:
                    pressed_o = True
                    print("O")
                if keys[pygame.K_p]:
                    pressed_p = True
                    print("P")
            elif event.type == pygame.KEYUP:
                #print("User let go of a key.")
                keys = pygame.key.get_pressed()
                
                if not keys[pygame.K_q]:
                    pressed_q = False
                    #print("!Q")
                if not keys[pygame.K_w]:
                    pressed_w = False
                    #print("!W")
                if not keys[pygame.K_o]:
                    pressed_o = False
                    #print("!O")
                if not keys[pygame.K_p]:
                    pressed_p = False
                    #print("!P")
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                print("User pressed a mouse button")
            elif event.type == pygame.MOUSEMOTION:
                #print("User moved the mouse")            
                po = pygame.mouse.get_pos()
                print(po)
            else:
                print("Other")
                
        # --- Logic
    
    
        l_bod = 1#self.body_length
        l_leg1 = 1#self.leg1_length
        l_leg2 = 1#self.leg2_length  
        
        a_bod = 0.5
        a_leg1 = 0.5
        a_leg2 = 0.5
        
        x= 0.00000000e+00
        y= 2.00000000e+00
        theta= 0 
        x1=2.50000000e-01
        y1=1.06698730e+00
        theta1=5.23598776e-01 
        x2=-2.50000000e-01 
        y2=1.06698730e+00
        theta2=-5.23598776e-01
        
        
        body_p = np.array([[x-(1-a_bod)*l_bod*np.sin(theta),y+(1-a_bod)*l_bod*np.cos(theta)],[x,y],[x+(a_bod)*l_bod*np.sin(theta),y-(a_bod)*l_bod*np.cos(theta)]])      
        leg1_p = np.array([[x1-(1-a_leg1)*l_leg1*np.sin(theta+theta1),y1+(1-a_leg1)*l_leg1*np.cos(theta+theta1)],[x1,y1],[x1+(a_leg1)*l_leg1*np.sin(theta+theta1),y1-(a_leg1)*l_leg1*np.cos(theta+theta1)]])
        leg2_p = np.array([[x2-(1-a_leg2)*l_leg2*np.sin(theta+theta2),y2+(1-a_leg2)*l_leg2*np.cos(theta+theta2)],[x2,y2],[x2+(a_leg2)*l_leg2*np.sin(theta+theta2),y2-(a_leg2)*l_leg2*np.cos(theta+theta2)]])      
    
        body_p[:,0]=(body_p[:,0]-x)*hum_scale+x_0
        leg1_p[:,0]=(leg1_p[:,0]-x)*hum_scale+x_0  
        leg2_p[:,0]=(leg2_p[:,0]-x)*hum_scale+x_0
        
        body_p[:,1]=y_0-(body_p[:,1])*hum_scale
        leg1_p[:,1]=y_0-(leg1_p[:,1])*hum_scale  
        leg2_p[:,1]=y_0-(leg2_p[:,1])*hum_scale
    
        
        
        # Determine location of painted lines based on X distance 
        
        # --- Drawing
        # Set the screen background (clears)
#        screen.fill(BLACK)
 #       screen.blit(bg_, (0, 0))
        screen.blit(background,(0,0))   
        #Draw text
        
        # generation                best x (gen)
        # time                      x
        n_trial = 13
        this_time = 3.245
        best_x = 20
        best_trial = 10
        this_x = 15
        
        # Select the font to use, size, bold, italics
        font = pygame.font.SysFont('courier', 18, True, False)
    
        text_time = font.render("%3.2fs" %this_time, True, WHITE)
        text_t_w = text_time.get_rect().width
        text_record = font.render("%3.2fm (%i)" % (best_x, best_trial), True, WHITE)
        
        screen.blit(text_record, [x_btn1,y_btn-dy_btn])
        screen.blit(text_time, [x_btn4+dx_btn-text_t_w,y_btn-dy_btn])
        
        
        font = pygame.font.SysFont('courier', 30, True, False)
        text_current = font.render("%3.2fm (%i)" %(this_x,n_trial), True, WHITE)
        text_c_w = text_current.get_rect().width
        text_c_h = text_current.get_rect().height
        
        screen.blit(text_current, [np.rint(width/2-text_c_w/2),y_btn])
        
        
        
        
        #Draw buttons (coloured or not)
        
        # Select the font to use, size, bold, italics
        font = pygame.font.SysFont('courier', 25, True, False)
     
        if pressed_q:
            pygame.draw.rect(screen, PRESSED, [x_btn1,y_btn,dx_btn,dy_btn])
            text_q = font.render("Q", True, TEXT_PRESSED)
            text_q_w = text_q.get_rect().width
            text_q_h = text_q.get_rect().height
            screen.blit(text_q, [x_btn1+np.rint(dx_btn/2-text_q_w/2), y_btn+np.rint(dy_btn/2-text_q_h/2)])
        if pressed_w:
            pygame.draw.rect(screen, PRESSED, [x_btn2,y_btn,dx_btn,dy_btn])
            text_w = font.render("W", True, TEXT_PRESSED)
            text_w_w = text_w.get_rect().width
            text_w_h = text_w.get_rect().height
            screen.blit(text_w, [x_btn2+np.rint(dx_btn/2-text_w_w/2), y_btn+np.rint(dy_btn/2-text_w_h/2)])
        if pressed_o:
            pygame.draw.rect(screen, PRESSED, [x_btn3,y_btn,dx_btn,dy_btn])
            text_o = font.render("O", True, TEXT_PRESSED)
            text_o_w = text_o.get_rect().width
            text_o_h = text_o.get_rect().height
            screen.blit(text_o, [x_btn3+np.rint(dx_btn/2-text_o_w/2), y_btn+np.rint(dy_btn/2-text_o_h/2)])
        if pressed_p:
            pygame.draw.rect(screen, PRESSED, [x_btn4,y_btn,dx_btn,dy_btn]) 
            text_p = font.render("P", True, TEXT_PRESSED)
            text_p_w = text_p.get_rect().width
            text_p_h = text_p.get_rect().height
            screen.blit(text_p, [x_btn4+np.rint(dx_btn/2-text_p_w/2), y_btn+np.rint(dy_btn/2-text_p_h/2)])
        
    
        
        #Draw body
        pygame.draw.line(screen, BODY, body_p[0], body_p[2], 10)
        pygame.draw.line(screen, BODY, leg1_p[0], leg1_p[2], 10)
        pygame.draw.line(screen, BODY, leg2_p[0], leg2_p[2], 10)
        
        pygame.draw.ellipse(screen, COG, [body_p[1][0]-5,body_p[1][1]-5,10,10], 2)
        pygame.draw.ellipse(screen, COG, [leg1_p[1][0]-5,leg1_p[1][1]-5,10,10], 2)
        pygame.draw.ellipse(screen, COG, [leg2_p[1][0]-5,leg2_p[1][1]-5,10,10], 2)
    
        # --- Wrap-up
        # Limit to 60 frames per second
        #clock.tick(60)
     
        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip() # more efficient: update(rectangle_list) https://www.pygame.org/docs/ref/display.html
        
     
    # Close everything down
    pygame.quit()
    
    print("quit")