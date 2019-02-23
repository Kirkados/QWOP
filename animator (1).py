#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:50:29 2019

@author: StephaneMagnan
"""

import numpy as np
import pygame
 
# Define dimensions for background

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

DIRT = (58,28,18)
TRACK = (208,69,40)
GRASS = (100,150,50)
SKY = (69,99,119)
SKY2 = (0,102,162)
LINE = (255,255,255)
STICK = (0,0,0)
GAMEOVER = (255,50,50)



#scale
hum_scale = 0.2 #pixel/mm
f_width = 800
f_height = 500


#define relative positions of background
y_sky = np.rint(0.5*f_height)
y_sky2 = np.rint(0.075*f_height)
y_grass = np.rint(0.175*f_height)
y_track = np.rint(0.2*f_height)
y_dirt = f_height-np.min((y_sky+y_sky2+y_grass+y_track,f_height))

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


#define relative positions of text
b_stats = np.rint(f_width-200)
b_mid = np.rint(f_width/2)

h_mid = np.rint(f_height/2)
h_record = np.rint(0.05*f_height)
h_update = np.rint(0.12*f_height)
h_prog = np.rint(0.95*f_height)

y_stats1 = np.rint(f_height/10)
y_stats2 = np.rint(2*f_height/10)
x_stats1 = np.rint(f_width/10)
x_stats2 = np.rint(8*f_width/10)

dx_btn = np.rint(f_height/12)
dy_btn = np.rint(f_height/12)

x_btn1 = np.rint(f_width/15)+20
x_btn2 = x_btn1+dx_btn+20
x_btn3 = f_width-x_btn2-dx_btn
x_btn4 =  f_width-x_btn1-dx_btn
y_btn = np.rint(1*f_height/10)







#define body dimensions
l_bod = 75
a_bod = 0.5
l_leg1 = 1
a_leg1 = 0.5
l_leg2= 1
a_leg2 = 0.5

#position
x_0 = np.rint(f_width/2)
y_0 = y_l3+np.rint((y_l4-y_l3)/2)

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
size = [f_width, f_height]
screen = pygame.display.set_mode(size)
 
 
# Loop until the user clicks the close button.
running = True
pressed_q = False
pressed_w = False
pressed_o = False
pressed_p = False
# Used to manage how fast the screen updates
#clock = pygame.time.Clock()



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
                print("!Q")
            if not keys[pygame.K_w]:
                pressed_w = False
                print("!W")
            if not keys[pygame.K_o]:
                pressed_o = False
                print("!O")
            if not keys[pygame.K_p]:
                pressed_p = False
                print("!P")
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            print("User pressed a mouse button")
        elif event.type == pygame.MOUSEMOTION:
            #print("User moved the mouse")            
            po = pygame.mouse.get_pos()
            print(po)
        else:
            print("other")
            
    # --- Logic

 
    
    # Determine location of painted lines based on X distance 
    
    # --- Drawing
    # Set the screen background (clears)
    screen.fill(BLACK)
 
    # Draw the background
    pygame.draw.rect(screen, SKY, [0,0,f_width,y_sky])
    pygame.draw.rect(screen, SKY2, [0,y_sky,f_width,y_sky2])
    pygame.draw.rect(screen, GRASS, [0,y_sky+y_sky2,f_width,y_grass])
    pygame.draw.rect(screen, TRACK, [0,y_sky+y_sky2+y_grass,f_width,y_track])
    pygame.draw.rect(screen, DIRT, [0,y_sky+y_sky2+y_grass+y_track,f_width,y_dirt])
    
    # Draw track lines
    pygame.draw.line(screen, LINE, [0, y_l1], [f_width, y_l1], t_l1)
    pygame.draw.line(screen, LINE, [0, y_l2], [f_width, y_l2], t_l2)
    pygame.draw.line(screen, LINE, [0, y_l3], [f_width, y_l3], t_l3)
    pygame.draw.line(screen, LINE, [0, y_l4], [f_width, y_l4], t_l4)
    
    
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
    
    screen.blit(text_current, [np.rint(f_width/2-text_c_w/2),y_btn])
    
    
    
    
    #Draw buttons (coloured or not)
    
    # Select the font to use, size, bold, italics
    font = pygame.font.SysFont('courier', 25, True, False)
 
    if pressed_q:
        text_q = font.render("Q", True, WHITE)
        text_q_w = text_q.get_rect().width
        text_q_h = text_q.get_rect().height
        pygame.draw.rect(screen, BLACK, [x_btn1,y_btn,dx_btn,dy_btn])
    else:
        text_q = font.render("Q", True, BLACK)
        text_q_w = text_q.get_rect().width
        text_q_h = text_q.get_rect().height
        pygame.draw.rect(screen, WHITE, [x_btn1,y_btn,dx_btn,dy_btn])
    if pressed_w:
        text_w = font.render("W", True, WHITE)
        text_w_w = text_w.get_rect().width
        text_w_h = text_w.get_rect().height
        pygame.draw.rect(screen, BLACK, [x_btn2,y_btn,dx_btn,dy_btn])
    else:
        text_w = font.render("W", True, BLACK)
        text_w_w = text_w.get_rect().width
        text_w_h = text_w.get_rect().height
        pygame.draw.rect(screen, WHITE, [x_btn2,y_btn,dx_btn,dy_btn])
    if pressed_o:
        text_o = font.render("O", True, WHITE)
        text_o_w = text_o.get_rect().width
        text_o_h = text_o.get_rect().height
        pygame.draw.rect(screen, BLACK, [x_btn3,y_btn,dx_btn,dy_btn])
    else:
        text_o = font.render("O", True, BLACK)
        text_o_w = text_o.get_rect().width
        text_o_h = text_o.get_rect().height
        pygame.draw.rect(screen, WHITE, [x_btn3,y_btn,dx_btn,dy_btn])
    if pressed_p:
        text_p = font.render("P", True, WHITE)
        text_p_w = text_p.get_rect().width
        text_p_h = text_p.get_rect().height
        pygame.draw.rect(screen, BLACK, [x_btn4,y_btn,dx_btn,dy_btn])
    else:
        text_p = font.render("P", True, BLACK)
        text_p_w = text_p.get_rect().width
        text_p_h = text_p.get_rect().height
        pygame.draw.rect(screen, WHITE, [x_btn4,y_btn,dx_btn,dy_btn])    
    
    # Put the image of the text on the screen at 250x250
    screen.blit(text_q, [x_btn1+np.rint(dx_btn/2-text_q_w/2), y_btn+np.rint(dy_btn/2-text_q_h/2)])
    screen.blit(text_w, [x_btn2+np.rint(dx_btn/2-text_w_w/2), y_btn+np.rint(dy_btn/2-text_w_h/2)])
    screen.blit(text_o, [x_btn3+np.rint(dx_btn/2-text_o_w/2), y_btn+np.rint(dy_btn/2-text_o_h/2)])
    screen.blit(text_p, [x_btn4+np.rint(dx_btn/2-text_p_w/2), y_btn+np.rint(dy_btn/2-text_p_h/2)])
    
    
    
    #Draw 
    

    # --- Wrap-up
    # Limit to 60 frames per second
    #clock.tick(60)
 
    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()
    
 
# Close everything down
pygame.quit()

print("quit")