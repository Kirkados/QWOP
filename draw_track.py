#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 08:21:10 2018

@author: StephaneMagnan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 07:47:06 2018

@author: StephaneMagnan
"""
import numpy as np
#import matlibplot as plt
from graphics import *

def line(x1,y1,x2,y2):
    return Line(Point(x1,y1),Point(x2,y2))

def trackMarks(x_cg,f_width,scale):
    #determine locations of track distance markers at the present time
    #uses centre of gravity to find middle, scale and f_width 
    #for spacing and bounds
    
    #x_cg = 2.4563
    #f_width = 800
    #scale = 0.2
    
    #number of unit steps in full screen
    f_steps = f_width/scale/1000
    #m increments in full screen
    f_inc = 1000*scale
       
    # find mid width
    mid = int(np.rint(f_width/2))
    
    # find the offset (left of centre for one mark)
    x_floor = int(np.floor(x_cg))
    offset = int((x_cg-np.floor(x_cg))*f_inc)
    
    tick_x = []
    tick_lab = []
        
        
    n_low = int(np.ceil((mid-offset)/f_inc))
    n_high = int(f_steps-n_low)
    
    
    for ind in range(n_low):
        tick_lab.append(x_floor-(ind))
        tick_x.append(mid-offset-(ind)*f_inc)
        
    for ind in range(n_high):
        tick_lab.append(x_floor+(ind+1))
        tick_x.append(mid-offset+(ind+1)*f_inc)     
    
    return tick_x, tick_lab




def main():

    x_cg = 4.635
    
    #pixel/mm
    hum_scale = 0.2
    f_width = 800
    f_height = 500
    
    
    #define relative positions of background
    b_stats = np.rint(f_width-200)
    b_mid = np.rint(f_width/2)
    
    h_mid = np.rint(f_height/2)
    h_record = np.rint(0.05*f_height)
    h_update = np.rint(0.12*f_height)
    h_prog = np.rint(0.95*f_height)
    h_track = np.rint(0.9*f_height)
    h_l1 = np.rint(0.75*f_height)
    h_l2 = np.rint(0.65*f_height)
    h_l3 = np.rint(0.59*f_height)
    h_grass = np.rint(0.55*f_height)
    h_sky = np.rint(0.35*f_height)
    
    
    c_dirt = color_rgb(58,28,18)
    c_track = color_rgb(208,69,40)
    c_grass = color_rgb(100,150,50)
    c_sky = color_rgb(0,102,162)
    c_line = color_rgb(255,255,255)
    c_stick = color_rgb(0,0,0)
    c_go = color_rgb(255,50,50)
    
    
    #draw background
    pt_s1 = Point(0,0)
    pt_s2 = Point(f_width,h_sky)
    
    pt_g1 = Point(0,h_sky)
    pt_g2 = Point(f_width,h_grass)    
    
    pt_t1 = Point(0,h_grass)
    pt_t2 = Point(f_width,h_track)
    
    pt_d1 = Point(0,h_track)
    pt_d2 = Point(f_width,f_height)  
     
    rect_dirt = Rectangle(pt_d1,pt_d2)
    rect_dirt.setFill(c_dirt)
    rect_dirt.setOutline(c_dirt)
    
    rect_track = Rectangle(pt_t1,pt_t2)
    rect_track.setFill(c_track)
    rect_track.setOutline(c_track)
    
    rect_grass = Rectangle(pt_g1,pt_g2)
    rect_grass.setFill(c_grass)
    rect_grass.setOutline(c_grass)
    
    rect_sky = Rectangle(pt_s1,pt_s2)
    rect_sky.setFill(c_sky)
    rect_sky.setOutline(c_sky)
    
    #define track lines
    pt_l11 = Point(0,h_grass)
    pt_l12 = Point(f_width,h_grass)
    pt_l21 = Point(0,h_l3)
    pt_l22 = Point(f_width,h_l3)
    pt_l31 = Point(0,h_l2)
    pt_l32 = Point(f_width,h_l2)
    pt_l41 = Point(0,h_l1)
    pt_l42 = Point(f_width,h_l1)
    pt_l51 = Point(0,h_track)
    pt_l52 = Point(f_width,h_track)
    
    lin_t1 = Line(pt_l11,pt_l12)
    lin_t1.setOutline(c_line)
    lin_t1.setWidth(2)
    
    lin_t2 = Line(pt_l21,pt_l22)
    lin_t2.setOutline(c_line)
    lin_t2.setWidth(3)
    
    lin_t3 = Line(pt_l31,pt_l32)
    lin_t3.setOutline(c_line)
    lin_t3.setWidth(4)
    
    lin_t4 = Line(pt_l41,pt_l42)
    lin_t4.setOutline(c_line)
    lin_t4.setWidth(5)
    
    lin_t5 = Line(pt_l51,pt_l52)
    lin_t5.setOutline(c_line)
    lin_t5.setWidth(6)
    
    
    #save static texts for later
    txt_go = Text(Point(b_mid,h_mid),"Game Over!")
    txt_go.setTextColor(c_go)
    txt_go.setFace('courier')
    txt_go.setSize(64)
    #txt_go.draw(win)
    
    
    # Initialize window
    win = GraphWin("QWOP", f_width, f_height)
    #NW pixel is (0,0)
    #win.setBackground(c_sky) 
    rect_dirt.draw(win)
    rect_track.draw(win)
    rect_grass.draw(win)
    rect_sky.draw(win)
    
    lin_t1.draw(win)
    lin_t2.draw(win)
    lin_t3.draw(win)
    lin_t4.draw(win)
    lin_t5.draw(win)
    
    
    best_m = 53.4
    best_t = 36.25
    best_ep = 16
    
    this_m = 26.1
    this_t = 42.5
    this_ep = 45
    
    #text for best/time/progress at top
    s_best = "Ep.%4i: %3.1f m  %3.1f s" %(best_ep,best_m, best_t)
    pt_best = Point(b_stats,h_record)
    txt_best = Text(pt_best,s_best)
    txt_best.setTextColor(c_line)
    txt_best.setSize(30)
    
    s_this = "Ep.%4i: %3.1f m  %3.1f s" %(this_ep,x_cg, this_t)
    pt_this = Point(b_stats,h_update)
    txt_curr = Text(pt_this,s_this)
    txt_curr.setTextColor(c_line)
    txt_curr.setSize(30)
    
    if best_m >=this_m:
        txt_best.setStyle("bold") 
        txt_curr.setStyle("normal") 
    else:
        txt_best.setStyle("normal") 
        txt_curr.setStyle("bold")  
        
    txt_best.draw(win)
    txt_curr.draw(win) 
    

        
    
    #text for progress at bottom
    tick_x, tick_lab = trackMarks(x_cg,f_width,hum_scale)
    for ptx in range(len(tick_x)):
        txt_prog = Text(Point(tick_x[ptx],h_prog),str(tick_lab[ptx]))
        txt_prog.setTextColor(c_line)
        txt_prog.setSize(24)
        txt_prog.draw(win)

    
        
    win.getMouse()
    win.close()
    
main()
    

def defCog(torso, arm_l, forearm_l, arm_r, forearm_r, thigh_l, calf_l, foot_l,thigh_r, calf_r, foot_r):
    print("blah")
    
    

def initHuman(joints, cogs, body, arm_l, forearm_l, arm_r, forearm_r, thigh_l, calf_l, foot_l, thigh_r, calf_r, foot_r):
    #define all points for drawing (store COGs and joints in different places)
    
    #origin at cog of H-T - assume distal towards head

    joints.shoulder[0] = 0
    joints.shoulder[1] = body.l*(1-body.cog)
    joints.hip[0] = 0
    joints.hip[1] = -body.l*(body.cog)
    
    cogs.body[0] = 0
    cogs.body[1] = 0    
    cogs.head[0] = 0
    cogs.head[1] = joints.shoulder[1]+(body.l_neck+body.d_head)*(1-body.cog_head)
    
    #arms are freely hanging
    joints.arm_l[0] = joints.shoulder[0]
    joints.arm_l[1] = joints.shoulder[1]-arm_l.l
    joints.forearm_l[0] = cogs.arm[0]
    joints.forearm_l[1] = cogs.arm[1]-forearm_l-l
  
    cogs.arm_l[0] = joints.shoulder[0]
    cogs.arm_l[1] = joints.shoulder[1]-arm_l.l*arm_l.cog
    cogs.forearm_l[0] = joints.forearm_l[0]
    cogs.forearm_l[1] = joints.forearm_l[1]-forearm_l.l*forearm_l.cog
    
    joints.arm_r[0] = cogs.arm_l[0]
    joints.arm_r[1] = cogs.arm_l[1] 
    joints.forearm_r[0] = cogs.forearm_l[0]
    joints.forearm_r[1] = cogs.forearm_l[1]
    
    cogs.arm_r[0] = cogs.arm_l[0]
    cogs.arm_r[1] = cogs.arm_l[1]
    cogs.forearm_r[0] = cogs.forearm_l[0]
    cogs.forearm_r[1] = cogs.forearm_l[1]

    #legs are dangling from body
    joints.knee_l[0] = joints.hip[0]
    joints.knee_l[1] = joints.hip[1] - thigh_l.l
    joints.ankle_l[0] = joints.knee_l[0]
    joints.ankle_l[1] = joints.knee_l[1] - calf_l.l
    joints.heel_l[0] = joints.ankle_l[0]
    joints.heel_l[1] = joints.ankle_l[1] - foot_l.l
    joints.toe_l[0] = joints.heel_l[0] + foot_l.w
    joints.toe_l[1] = joints.heel_l[1]  

    cogs.knee_l[0] = joints.hip[0]
    cogs.knee_l[1] = joints.hip[1] - thigh_l.l*thigh_l.cog
    cogs.calf_l[0] = joints.knee_l[0]
    cogs.calf_l[1] = joints.knee_l[1] - calf_l.l*calf_l.cog
    cogs.foot_l[0] = joints.ankle_l[0]+foot_l.w*foot_l.cog
    cogs.foot_l[1] = joints.ankle_l[1]-foot_l.l*foot_l.cog
    
    joints.knee_r[0] = joints.knee_l[0]
    joints.knee_r[1] = joints.knee_l[1]
    joints.ankle_r[0] = joints.ankle_l[0]
    joints.ankle_r[1] = joints.ankle_l[1]
    joints.heel_r[0] = joints.heel_l[0]
    joints.heel_r[1] = joints.heel_l[1]
    joints.toe_r[0] = joints.toe_l[0]
    joints.toe_r[1] = joints.toe_l[1]    

    cogs.knee_r[0] = cogs.knee_l[0]
    cogs.knee_r[1] = cogs.knee_l[1]
    cogs.calf_r[0] = cogs.calf_l[0]
    cogs.calf_r[1] = cogs.calf_l[1]
    cogs.foot_r[0] = cogs.foot_l[0]
    cogs.foot_r[1] = cogs.foot_l[1] 

    #apply rotations to all points
    
    
    #apply global translation
    
    return joints, cogs 

def transHuman(thetas,initHuman):
    #first apply rotations
    print("blah")
    
    
    
    
    

class COGs():
    def __init__(self,def_l,def_cog,def_i,init_x,init_y,init_t):
        self.body = [0,0]
        self.head = [0,0]
        
        self.thigh_l = [0,0]
        self.thigh_r = [0,0]
        self.calf_l = [0,0]
        self.calf_r = [0,0]
        self.foot_l = [0,0]
        self.foot_r = [0,0]
        
        self.arm_l = [0,0]
        self.arm_r = [0,0]
        self.forearm_l = [0,0]
        self.forearm_r = [0,0]
        
        return
    
class Joints():
    def __init__(self,def_l,def_cog,def_i,init_x,init_y,init_t):
        #x_coord, y_coord
        self.shoulder = [0,0]
        self.head = [0,0]
        self.hip = [0,0]
        
        self.knee_l = [0,0]
        self.knee_r = [0,0]
        self.ankle_l = [0,0]
        self.ankle_r = [0,0] 
        self.heel_l = [0,0]
        self.heel_r = [0,0]
        self.toe_l = [0,0]
        self.toe_r = [0,0]
        
        self.elbow_l = [0,0]
        self.elbow_r = [0,0]
        self.hand_l = [0,0]
        self.hand_r = [0,0]
        
        return
    
    

    
class Body():
    #includes head, neck and torso (no arms or legs)
      def __init__(self,def_l,def_cog,def_i,init_x,init_y,init_t):
        self.l = def_l
        self.l_neck = 0
        self.d_head = 0
        
        #H-T cog and i.
        self.cog = def_cog
        self.i = def_i
        self.cog_head = 0
        
        self.x = init_x
        self.y = init_y
        self.t = init_t
        
        self.x_prox = 0
        self.y_prox = 0
        self.x_dist = 0
        self.y_dist = 0  
 
        self.dx = 0
        self.dy = 0
        self.dt = 0
        
        self.ddx = 0
        self.ddy = 0
        self.ddt = 0
        
        return
    
    def state(self):
        return self.x, self.y, self.t  
    
class Segment():
    def __init__(self,def_l,def_cog,def_i,init_x,init_y,init_t):
        self.l = def_l
        self.h = 0 #height only used for foot
        self.cog = def_cog
        self.i = def_i
        
        self.x = init_x
        self.y = init_y
        self.t = init_t
        
        self.x_prox = 0
        self.y_prox = 0
        self.x_dist = 0
        self.y_dist = 0  
 
        self.dx = 0
        self.dy = 0
        self.dt = 0
        
        self.ddx = 0
        self.ddy = 0
        self.ddt = 0
        
        return
    
#    def X(self, x_state):
#        self.x = x_state
        
    def state(self):
        return self.x, self.y, self.t



class Body():
'body = Segment(play_height,play_mass,hip2shoul_len, hip2shoul_com,should2headc_len,should2headc_com,head_diam,hip2shoul_k,hip2shoul_m, should2headc_k,should2headc_m)
    def __init__(self,play_height,play_mass,def_leng1,def_beta1,def_leng2,def_beta2,def_diam,def_rad1,def_mass1,def_rad2,def_mass2):
        
        'segment height hip-shoulder
        self.length1 = def_leng1*play_height
        self.beta1 = def_beta1*self.length1
        
                
        self.radgyr1 = def_rad1*(self.length1)
        self.mass1 = def_mass1*play_mass
        self.momin1 = self.mass1*self.radgyr1**2
        
        'segment height shoulder-centre of head
        self.length2 = def_leng2*play_height
        'head diameter
        self.diam = def_diam*play_height       
        'head/neck com based on neck-head centre  length
        self.beta2 = def_beta2*(self.length2+self.diam/2)/self.length2
        
        self.radgyr2 = def_rad2*(self.length2+self.diam/2)
        self.mass2 = def_mass2*play_mass
        self.momin2 = self.mass2*self.radgyr2**2
        
        'body definitions
        self.length = (def_leng1+def_leng2+def_diam/2)*play_height
        self.beta = (self.beta1*self.length1*self.mass1+(self.length1+self.beta2*self.length2*self.mass2))/(self.mass1+self.mass2)/self.length
        
        self.momin = self.momin1+self.mass1*(self.beta*self.length-self.beta1*self.length1)**2+self.momin2+self.mass2*(self.length1+self.beta2*self.length2-self.beta*self.length)**2
        
        return
    
#    def X(self, x_state):
#        self.x = x_state
        
    def state(self):
        return self.length, self.beta, self.radgyr, self.mass, self.momin 
    

class Segment():
'arm_prox = Segment(play_height,play_weight,seg_length,seg_cog,seg_radgry,seg_mass)
    def __init__(self,play_height, play_mass, def_leng,def_beta,def_rad,def_mass):
        
        'segment height
        self.length = def_leng*play_height
        self.beta = def_beta 
        
        'radius, mass, moment of inertia
        self.radgyr = def_rad*self.length
        self.mass = def_mass*play_mass
        self.momin = self.mass*self.radgyr**2
        
        return
    
#    def X(self, x_state):
#        self.x = x_state
        
    def state(self):
        return self.length, self.beta, self.radgyr, self.mass, self.momin 
    
    
class Foot():
'foot = Foot(play_height,play_weight,foot_height,foot_cog_h,foot_length,foot_cog_l,foot_radgry,foot_mass)
    def __init__(self,play_height, play_mass,def_leng1,def_beta1,def_leng2,def_beta2,def_rad,def_mass):
        
        'foot height
        self.length1 = def_leng1*play_height
        self.beta1 = def_beta1
        'foot length
        self.length2 = def_leng2*play_height
        self.beta2 = def_beta2
        'radius, mass, moment of inertia
        self.radgyr = def_rad*self.length2
        self.mass = def_mass*play_mass
        self.momin = self.mass*self.radgyr**2
        
        return
    
#    def X(self, x_state):
#        self.x = x_state
        
    def state(self):
        return self.length1, self.beta1, self.length2, self.beta2, self.radgyr, self.mass, self.momin 