#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:23:53 2019

@author: StephaneMagnan
"""

import pygame
from pygame.locals import *
yellow = (255,255,0)    # RGB color tuple
    
# initialise screen
pygame.init()
screen = pygame.display.set_mode((350, 250))
pygame.display.set_caption('Basic Pygame program')
# fill background
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill(yellow)
# display some text
font = pygame.font.Font(None, 36)
text = font.render("Hello from Monty PyGame", 1, (10, 10, 10))
textpos = text.get_rect()
textpos.centerx = background.get_rect().centerx
background.blit(text, textpos)
# blit everything to the screen
screen.blit(background, (0, 0))
pygame.display.flip()
# event loop
while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("triggered")
            break
        else:
            print("no")
    screen.blit(background, (0, 0))
    pygame.display.flip()
pygame.quit()    
print("end")