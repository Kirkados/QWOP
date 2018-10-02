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

def main():

    f_width = 800
    f_height = 500
    
    
    #define relative positions of background
    b_up1 = np.rint(100)
    b_up2 = np.rint(f_width-100)
    b_mid = np.rint(f_width/2)
    
    h_update = np.rint(0.05*f_height)
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
    
    this_m = 26.1
    this_t = 42.5
    
    #text for best/time/progress at top
    s_best = "%3.1f m  %3.1f s" %(best_m, best_t)
    pt_best = Point(b_up1,h_update)
    txt_best = Text(pt_best,s_best)
    txt_best.setTextColor(c_line)
    txt_best.setSize(30)
    txt_best.draw(win)
     
    s_this = "%3.1f m  %3.1f s" %(this_m, this_t)
    pt_this = Point(b_up2,h_update)
    txt_curr = Text(pt_this,s_this)
    txt_curr.setTextColor(c_line)
    txt_curr.setSize(30)
    txt_curr.draw(win)     
    
    #text for progress at bottom
    txt_prog = Text(Point(b_mid,h_prog),"0")
    txt_prog.setTextColor(c_line)
    txt_prog.setSize(24)
    txt_prog.draw(win)
    
    
        
    win.getMouse()
    win.close()
    
main()
    