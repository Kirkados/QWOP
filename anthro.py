#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:57:41 2018

@author: StephaneMagnan
"""

play_height = 1.8'm
play_mass = 75'kg

'foot
h_foot = 0.43
l_foot = 0.152
comh_foot = 0.500
com_foot = 0.500
k_foot = 0.475
m_foot = 0.0145

'leg distal
l_leg_dist = 0.242
com_leg_dist = 0.433
k_leg_dist = 0.302
m_leg_dist = 0.465

'leg proximal
l_leg_prox = 0.245
com_leg_prox = 0.433
k_leg_prox = 0.323
m_leg_prox = 0.100

'arm distal
l_arm_dist = 0.145
com_arm_dist = 0.430
k_arm_dist = 0.303
m_arm_dist = 0.016

'arm proximal
l_arm_prox = 0.189
com_arm_prox = 0.530
k_arm_prox = 0.368
m_arm_prox = 0.028




'foot = Foot(play_height,play_weight,foot_height,foot_cog_h,foot_length,foot_cog_l,foot_radgry,foot_mass)
foot = Foot(play_height,play_mass,h_foot,comh_foot,l_foot,com_foot,k_foot,m_foot)

'arm_prox = Segment(play_height,play_mass,seg_length,seg_cog,seg_radgry,seg_mass)
leg_distal = Segment(play_height,play_mass,l_leg_dist,com_leg_dist,k_leg_dist,m_leg_dist)
leg_proximal = Segment(play_height,play_mass,l_leg_prox,com_leg_prox,k_leg_prox,m_leg_prox)
arm_distal = Segment(play_height,play_mass,l_arm_dist,com_arm_dist,k_arm_dist,m_arm_dist)
arm_proximal = Segment(play_height,play_mass,l_arm_prox,com_arm_prox,k_arm_prox,m_arm_prox)


body = Body(play_height,play_mass,)




'player stats
p_mass = 75'kg
p_height = 180'cm

'define components
'from http://health.uottawa.ca/biomech/courses/apa2313/bsptable.pdf
'from http://www.oandplibrary.org/al/pdf/1972_01_001.pdf
'from http://jestec.taylors.edu.my/Vol%2011%20issue%202%20February%202016/Volume%20(11)%20Issue%20(2)%20166-%20176.pdf




'foot
m_foot = 0.0145*p_mass
h_foot = 0.43*p_height
l_foot = 0.152*p_height
comh_foot = 0.500*h_foot
com_foot = 0.500*l_foot
k_foot = 0.475*l_foot
ic_foot = m_foot*k_foot**2
ip_foot = m_foot*com_foot**2

'leg_dist
m_leg_dist = 0.465*p_mass
l_leg_dist = 0.242*p_height
com_leg_dist = 0.433*l_leg_dist
k_leg_dist = 0.302*l_leg_dist
ic_leg_dist = m_leg_dist*k_leg_dist**2
ip_leg_dist = m_leg_dist*com_leg_dist**2

'thigh
m_thigh = 0.100*p_mass
l_thigh = 0.245*p_height
com_thigh = 0.433*l_thigh
k_thigh = 0.323*l_thigh
ic_thigh = m_thigh*k_thigh**2
ip_thigh = m_thigh*com_thigh**2

'hand
m_hand = 0.006*p_mass
l_hand = 0.128*p_height
com_hand = 0.506*l_hand
k_hand = 0.297*l_hand
ic_hand = m_hand*k_hand**2
ip_hand = m_hand*com_hand**2

'forearm
m_forearm = 0.016*p_mass
l_forearm = 0.145*p_height
com_forearm = 0.430*l_forearm
k_forearm = 0.303*l_forearm
ic_forearm = m_forearm*k_forearm**2
ip_forearm = m_forearm*com_forearm**2

'arm
m_arm = 0.028*p_mass
l_arm = 0.189*p_height
com_arm = 0.530*l_arm
k_arm = 0.368*l_arm
ic_arm = m_arm*k_arm**2
ip_arm = m_arm*com_arm**2

'trunk
m_trunk = 0.497*p_mass
l_trunk = 0.289*p_height
com_trunk = 0.500*l_trunk
k_trunk = 0.500*l_trunk
ic_trunk = m_trunk*k_trunk**2
ip_trunk = m_trunk*com_trunk**2

'head+neck
m_head = 0.181*p_mass
l_head = 0.19*p_height
com_head = 0.567*l_head
k_head = 0.495*l_head
ic_head = m_head*k_head**2
ip_head = m_head*com_head**2