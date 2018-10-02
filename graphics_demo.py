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
    win = GraphWin("QWOP", f_width, f_height)
    #NW pixel is (0,0)
    win.setBackground(color_rgb(255,0,0))
    
    
    pt1 = Point(100,50)
    pt2 = Point(275,465)
    pt3 = Point(38,12)
    pt4 = Point(300,250)
    
    pt1.setOutline(color_rgb(100,0,200))
    pt1.draw(win)
    
    rad = 50
    cir = Circle(pt4,rad)
    cir.setFill(color_rgb(100,100,10))
    cir.draw(win)
    
    
    lin = Line(pt1,pt2)
    lin.setOutline(color_rgb(120,0,120))
    lin.setWidth(5)
    lin.draw(win)
    
    rect = Rectangle(pt3,pt1)
    rect.setOutline(color_rgb(20,255,100))
    rect.setFill(color_rgb(120,255,200))
    rect.setWidth(5)
    rect.draw(win)
    
    poly = Polygon(pt1,pt2,pt3)
    poly.setOutline(color_rgb(20,255,100))
    poly.setFill(color_rgb(120,255,200))
    poly.setWidth(5)
    poly.draw(win)
    
    txt = Text(Point(150,150),"Hello World!)
    txt.setTextColor(color_rgb(255,255,255))
    txt.setSize(30)
    txt.setFace('Courier')
    txt.show(win)
    
    win.getMouse()
    win.close()
    
main()
    