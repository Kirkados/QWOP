# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 14:57:47 2019

Checks what the mass matrix is at any state

@author: Kirk
"""

import environment_qwop_full_11
import numpy as np

env = environment_qwop_full_11.Environment()

#state = [1.0660072464299022,   0.30319930780076043,   0.7103450582049328,   0.3472918303670004,   2.475997771802081   ,2.216665473808854   ,2.315823868288919   ,0.8972503211948798   ,1.5868585164637663   ,-0.5383956806940552   ,-2.0507399475110186   ,3.031277085356769   ,3.6705520485751517   ,-17.356535047334468   ,-18.099546484930435   ,-18.338724596261148   ,-22.490812166704103   ,-27.227927498008526   ,-20.278860643254372   ,-24.70003577240068   ,-4.449719340128093   ,3.1926632780894866] # mid broken
#state = [0.48314363036792457,   2.0616865550000685  , 0.6086610813540608 ,  0.31364511805075596  , 1.6948438780227415 ,  1.9343639150383194,   2.7256379908386914 ,  0.9053836056202855 ,  2.4004156561182346 ,  -0.03283845544102466  , -1.7926507803707608 ,  0.9072732457142894  , -0.9927362130211982  , 0.7555432249170714 ,  0.5358956966539228 ,  0.36054091490930046 ,  2.6281155955241284 ,  3.4215623300011817 ,  0.4150179590166816  , 0.2367945045317151 ,  0.06748934595922974 ,  -0.07438020074749815] # 5 before broken
state = [0.9621638225527204 ,  0.33102127622152805  , 1.0800466724625302 ,  0.48171384872954814 ,  1.9672905897326982 ,  2.7637257532462556  , 3.477185959582926 ,  1.6544366833455897  , 3.0522669458812923  , -0.05583322601268653 ,  -1.9744046936851094 ,  1.9490712587725993  , -3.468890462692185 ,  -3.4858275150437894  , -6.25923652963948  , -7.21020507721745  , -0.23661757702902497  , 0.48713361434560604  , -0.28769109209545846 ,  0.2600484865937872  , -8.246318594686219  , -8.035719306244472] # first broken

#angles = [-0.0872664625997165,   2.2689280275926285,   1.1344640137963142 ,  -0.08726646259971656  , 0.0872664625997165   ,0.6108652381980152   ,-0.4363323129985822  , -0.8726646259971647] # mid broken
#angles = [-0.2617993877991495 ,  1.3962634015954647 ,  1.3089969389957468 ,  0.7853981633974483 ,  0.2617993877991495 ,  1.4835298641951795  , -0.6108652381980153 ,  -1.7453292519943286] # 5 before broken
angles = [-0.6108652381980154 ,  1.483529864195181 ,  1.6580627893946123 ,  0.6981317007977318,   0.6108652381980154 ,  1.3962634015954631  , -0.9599310885968811 ,  -1.6580627893946123] # first broken
phi1r, phi2r, phi3r, phi4r, phi1l, phi2l, phi3l, phi4l = angles


# Unpacking the state
x,       y,    theta,    theta1r,    theta2r,    theta3r,    theta4r,    theta1l,    theta2l,    theta3l,    theta4l, \
xdot, ydot, thetadot, theta1rdot, theta2rdot, theta3rdot, theta4rdot, theta1ldot, theta2ldot, theta3ldot, theta4ldot = state

parameters = np.array([env.SEGMENT_MASS[0], env.SEGMENT_MASS[1], env.SEGMENT_MASS[2], env.SEGMENT_MASS[3], env.SEGMENT_MASS[4], \
                               env.SEGMENT_LENGTH[0], env.SEGMENT_LENGTH[1], env.SEGMENT_LENGTH[2], env.SEGMENT_LENGTH[3], env.SEGMENT_LENGTH[4], \
                               env.SEGMENT_GAMMA_LENGTH[0], env.SEGMENT_GAMMA_LENGTH[1], env.SEGMENT_GAMMA_LENGTH[2], env.SEGMENT_GAMMA_LENGTH[3], env.SEGMENT_GAMMA_LENGTH[4], \
                               env.SEGMENT_MOMINERT[0], env.SEGMENT_MOMINERT[1], env.SEGMENT_MOMINERT[2], env.SEGMENT_MOMINERT[3], env.SEGMENT_MOMINERT[4], \
                               env.HIP_SPRING_STIFFNESS, env.HIP_DAMPING, env.CALF_SPRING_STIFFNESS, env.CALF_DAMPING, \
                               env.SHOULDER_SPRING_STIFFNESS, env.SHOULDER_DAMPING, env.ELBOW_SPRING_STIFFNESS, env.ELBOW_DAMPING, \
                               env.g, env.FLOOR_SPRING_STIFFNESS, env.FLOOR_DAMPING_COEFFICIENT, env.FLOOR_FRICTION_STIFFNESS, env.FLOOR_MU, env.ATANDEL, env.ATANGAIN, \
                               phi1r, phi2r, phi3r, phi4r, phi1l, phi2l, phi3l, phi4l], dtype = 'float64')



# Unpacking parameters
m, m1, m2, m3, m4, \
l, l1, l2, l3, l4, \
gamma, gamma1, gamma2, gamma3, gamma4,\
I, I1, I2, I3, I4, \
k3, c3, k4, c4, \
k1, c1, k2, c2, \
g, FLOOR_SPRING_STIFFNESS, FLOOR_DAMPING_COEFFICIENT, FLOOR_FRICTION_STIFFNESS, FLOOR_MU, ATANDEL, ATANGAIN, \
phi1r, phi2r, phi3r, phi4r, phi1l, phi2l, phi3l, phi4l = parameters

M = np.matrix([[	m + 2*m1 + 2*m2 + 2*m3 + 2*m4	,	0	,	-2*gamma*m1*np.cos(theta) - 2*gamma*m2*np.cos(theta) - 2*gamma*m3*np.cos(theta) - 2*gamma*m4*np.cos(theta) + 2*l*m3*np.cos(theta) + 2*l*m4*np.cos(theta)	,	gamma1*m1*np.cos(theta1r) + l1*m2*np.cos(theta1r)	,	gamma2*m2*np.cos(theta2r)	,	gamma3*m3*np.cos(theta3r) + l3*m4*np.cos(theta3r)	,	gamma4*m4*np.cos(theta4r)	,	gamma1*m1*np.cos(theta1l) + l1*m2*np.cos(theta1l)	,	gamma2*m2*np.cos(theta2l)	,	gamma3*m3*np.cos(theta3l) + l3*m4*np.cos(theta3l)	,	gamma4*m4*np.cos(theta4l)	],
                    [	0	,	m + 2*m1 + 2*m2 + 2*m3 + 2*m4	,	-2*gamma*m1*np.sin(theta) - 2*gamma*m2*np.sin(theta) - 2*gamma*m3*np.sin(theta) - 2*gamma*m4*np.sin(theta) + 2*l*m3*np.sin(theta) + 2*l*m4*np.sin(theta)	,	gamma1*m1*np.sin(theta1r) + l1*m2*np.sin(theta1r)	,	gamma2*m2*np.sin(theta2r)	,	gamma3*m3*np.sin(theta3r) + l3*m4*np.sin(theta3r)	,	gamma4*m4*np.sin(theta4r)	,	gamma1*m1*np.sin(theta1l) + l1*m2*np.sin(theta1l)	,	gamma2*m2*np.sin(theta2l)	,	gamma3*m3*np.sin(theta3l) + l3*m4*np.sin(theta3l)	,	gamma4*m4*np.sin(theta4l)	],
                    [	-2*gamma*m1*np.cos(theta) - 2*gamma*m2*np.cos(theta) - 2*gamma*m3*np.cos(theta) - 2*gamma*m4*np.cos(theta) + 2*l*m3*np.cos(theta) + 2*l*m4*np.cos(theta)	,	-2*gamma*m1*np.sin(theta) - 2*gamma*m2*np.sin(theta) - 2*gamma*m3*np.sin(theta) - 2*gamma*m4*np.sin(theta) + 2*l*m3*np.sin(theta) + 2*l*m4*np.sin(theta)	,	I + 2*gamma**2*m1*np.sin(theta)**2 + 2*gamma**2*m1*np.cos(theta)**2 + 2*gamma**2*m2*np.sin(theta)**2 + 2*gamma**2*m2*np.cos(theta)**2 + 2*gamma**2*m3*np.sin(theta)**2 + 2*gamma**2*m3*np.cos(theta)**2 + 2*gamma**2*m4*np.sin(theta)**2 + 2*gamma**2*m4*np.cos(theta)**2 - 4*gamma*l*m3*np.sin(theta)**2 - 4*gamma*l*m3*np.cos(theta)**2 - 4*gamma*l*m4*np.sin(theta)**2 - 4*gamma*l*m4*np.cos(theta)**2 + 2*l**2*m3*np.sin(theta)**2 + 2*l**2*m3*np.cos(theta)**2 + 2*l**2*m4*np.sin(theta)**2 + 2*l**2*m4*np.cos(theta)**2	,	-gamma*gamma1*m1*np.sin(theta)*np.sin(theta1r) - gamma*gamma1*m1*np.cos(theta)*np.cos(theta1r) - gamma*l1*m2*np.sin(theta)*np.sin(theta1r) - gamma*l1*m2*np.cos(theta)*np.cos(theta1r)	,	-gamma*gamma2*m2*np.sin(theta)*np.sin(theta2r) - gamma*gamma2*m2*np.cos(theta)*np.cos(theta2r)	,	-gamma*gamma3*m3*np.sin(theta)*np.sin(theta3r) - gamma*gamma3*m3*np.cos(theta)*np.cos(theta3r) - gamma*l3*m4*np.sin(theta)*np.sin(theta3r) - gamma*l3*m4*np.cos(theta)*np.cos(theta3r) + gamma3*l*m3*np.sin(theta)*np.sin(theta3r) + gamma3*l*m3*np.cos(theta)*np.cos(theta3r) + l*l3*m4*np.sin(theta)*np.sin(theta3r) + l*l3*m4*np.cos(theta)*np.cos(theta3r)	,	-gamma*gamma4*m4*np.sin(theta)*np.sin(theta4r) - gamma*gamma4*m4*np.cos(theta)*np.cos(theta4r) + gamma4*l*m4*np.sin(theta)*np.sin(theta4r) + gamma4*l*m4*np.cos(theta)*np.cos(theta4r)	,	-gamma*gamma1*m1*np.sin(theta)*np.sin(theta1l) - gamma*gamma1*m1*np.cos(theta)*np.cos(theta1l) - gamma*l1*m2*np.sin(theta)*np.sin(theta1l) - gamma*l1*m2*np.cos(theta)*np.cos(theta1l)	,	-gamma*gamma2*m2*np.sin(theta)*np.sin(theta2l) - gamma*gamma2*m2*np.cos(theta)*np.cos(theta2l)	,	-gamma*gamma3*m3*np.sin(theta)*np.sin(theta3l) - gamma*gamma3*m3*np.cos(theta)*np.cos(theta3l) - gamma*l3*m4*np.sin(theta)*np.sin(theta3l) - gamma*l3*m4*np.cos(theta)*np.cos(theta3l) + gamma3*l*m3*np.sin(theta)*np.sin(theta3l) + gamma3*l*m3*np.cos(theta)*np.cos(theta3l) + l*l3*m4*np.sin(theta)*np.sin(theta3l) + l*l3*m4*np.cos(theta)*np.cos(theta3l)	,	-gamma*gamma4*m4*np.sin(theta)*np.sin(theta4l) - gamma*gamma4*m4*np.cos(theta)*np.cos(theta4l) + gamma4*l*m4*np.sin(theta)*np.sin(theta4l) + gamma4*l*m4*np.cos(theta)*np.cos(theta4l)	],
                    [	gamma1*m1*np.cos(theta1r) + l1*m2*np.cos(theta1r)	,	gamma1*m1*np.sin(theta1r) + l1*m2*np.sin(theta1r)	,	-gamma*gamma1*m1*np.sin(theta)*np.sin(theta1r) - gamma*gamma1*m1*np.cos(theta)*np.cos(theta1r) - gamma*l1*m2*np.sin(theta)*np.sin(theta1r) - gamma*l1*m2*np.cos(theta)*np.cos(theta1r)	,	I1 + gamma1**2*m1*np.sin(theta1r)**2 + gamma1**2*m1*np.cos(theta1r)**2 + l1**2*m2*np.sin(theta1r)**2 + l1**2*m2*np.cos(theta1r)**2	,	gamma2*l1*m2*np.sin(theta1r)*np.sin(theta2r) + gamma2*l1*m2*np.cos(theta1r)*np.cos(theta2r)	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	gamma2*m2*np.cos(theta2r)	,	gamma2*m2*np.sin(theta2r)	,	-gamma*gamma2*m2*np.sin(theta)*np.sin(theta2r) - gamma*gamma2*m2*np.cos(theta)*np.cos(theta2r)	,	gamma2*l1*m2*np.sin(theta1r)*np.sin(theta2r) + gamma2*l1*m2*np.cos(theta1r)*np.cos(theta2r)	,	I2 + gamma2**2*m2*np.sin(theta2r)**2 + gamma2**2*m2*np.cos(theta2r)**2	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	gamma3*m3*np.cos(theta3r) + l3*m4*np.cos(theta3r)	,	gamma3*m3*np.sin(theta3r) + l3*m4*np.sin(theta3r)	,	-gamma*gamma3*m3*np.sin(theta)*np.sin(theta3r) - gamma*gamma3*m3*np.cos(theta)*np.cos(theta3r) - gamma*l3*m4*np.sin(theta)*np.sin(theta3r) - gamma*l3*m4*np.cos(theta)*np.cos(theta3r) + gamma3*l*m3*np.sin(theta)*np.sin(theta3r) + gamma3*l*m3*np.cos(theta)*np.cos(theta3r) + l*l3*m4*np.sin(theta)*np.sin(theta3r) + l*l3*m4*np.cos(theta)*np.cos(theta3r)	,	0	,	0	,	I3 + gamma3**2*m3*np.sin(theta3r)**2 + gamma3**2*m3*np.cos(theta3r)**2 + l3**2*m4*np.sin(theta3r)**2 + l3**2*m4*np.cos(theta3r)**2	,	gamma4*l3*m4*np.sin(theta3r)*np.sin(theta4r) + gamma4*l3*m4*np.cos(theta3r)*np.cos(theta4r)	,	0	,	0	,	0	,	0	],
                    [	gamma4*m4*np.cos(theta4r)	,	gamma4*m4*np.sin(theta4r)	,	-gamma*gamma4*m4*np.sin(theta)*np.sin(theta4r) - gamma*gamma4*m4*np.cos(theta)*np.cos(theta4r) + gamma4*l*m4*np.sin(theta)*np.sin(theta4r) + gamma4*l*m4*np.cos(theta)*np.cos(theta4r)	,	0	,	0	,	gamma4*l3*m4*np.sin(theta3r)*np.sin(theta4r) + gamma4*l3*m4*np.cos(theta3r)*np.cos(theta4r)	,	I4 + gamma4**2*m4*np.sin(theta4r)**2 + gamma4**2*m4*np.cos(theta4r)**2	,	0	,	0	,	0	,	0	],
                    [	gamma1*m1*np.cos(theta1l) + l1*m2*np.cos(theta1l)	,	gamma1*m1*np.sin(theta1l) + l1*m2*np.sin(theta1l)	,	-gamma*gamma1*m1*np.sin(theta)*np.sin(theta1l) - gamma*gamma1*m1*np.cos(theta)*np.cos(theta1l) - gamma*l1*m2*np.sin(theta)*np.sin(theta1l) - gamma*l1*m2*np.cos(theta)*np.cos(theta1l)	,	0	,	0	,	0	,	0	,	I1 + gamma1**2*m1*np.sin(theta1l)**2 + gamma1**2*m1*np.cos(theta1l)**2 + l1**2*m2*np.sin(theta1l)**2 + l1**2*m2*np.cos(theta1l)**2	,	gamma2*l1*m2*np.sin(theta1l)*np.sin(theta2l) + gamma2*l1*m2*np.cos(theta1l)*np.cos(theta2l)	,	0	,	0	],
                    [	gamma2*m2*np.cos(theta2l)	,	gamma2*m2*np.sin(theta2l)	,	-gamma*gamma2*m2*np.sin(theta)*np.sin(theta2l) - gamma*gamma2*m2*np.cos(theta)*np.cos(theta2l)	,	0	,	0	,	0	,	0	,	gamma2*l1*m2*np.sin(theta1l)*np.sin(theta2l) + gamma2*l1*m2*np.cos(theta1l)*np.cos(theta2l)	,	I2 + gamma2**2*m2*np.sin(theta2l)**2 + gamma2**2*m2*np.cos(theta2l)**2	,	0	,	0	],
                    [	gamma3*m3*np.cos(theta3l) + l3*m4*np.cos(theta3l)	,	gamma3*m3*np.sin(theta3l) + l3*m4*np.sin(theta3l)	,	-gamma*gamma3*m3*np.sin(theta)*np.sin(theta3l) - gamma*gamma3*m3*np.cos(theta)*np.cos(theta3l) - gamma*l3*m4*np.sin(theta)*np.sin(theta3l) - gamma*l3*m4*np.cos(theta)*np.cos(theta3l) + gamma3*l*m3*np.sin(theta)*np.sin(theta3l) + gamma3*l*m3*np.cos(theta)*np.cos(theta3l) + l*l3*m4*np.sin(theta)*np.sin(theta3l) + l*l3*m4*np.cos(theta)*np.cos(theta3l)	,	0	,	0	,	0	,	0	,	0	,	0	,	I3 + gamma3**2*m3*np.sin(theta3l)**2 + gamma3**2*m3*np.cos(theta3l)**2 + l3**2*m4*np.sin(theta3l)**2 + l3**2*m4*np.cos(theta3l)**2	,	gamma4*l3*m4*np.sin(theta3l)*np.sin(theta4l) + gamma4*l3*m4*np.cos(theta3l)*np.cos(theta4l)	],
                    [	gamma4*m4*np.cos(theta4l)	,	gamma4*m4*np.sin(theta4l)	,	-gamma*gamma4*m4*np.sin(theta)*np.sin(theta4l) - gamma*gamma4*m4*np.cos(theta)*np.cos(theta4l) + gamma4*l*m4*np.sin(theta)*np.sin(theta4l) + gamma4*l*m4*np.cos(theta)*np.cos(theta4l)	,	0	,	0	,	0	,	0	,	0	,	0	,	gamma4*l3*m4*np.sin(theta3l)*np.sin(theta4l) + gamma4*l3*m4*np.cos(theta3l)*np.cos(theta4l)	,	I4 + gamma4**2*m4*np.sin(theta4l)**2 + gamma4**2*m4*np.cos(theta4l)**2	]])

# Calculating lower limb positions and velocities
x4rdot = xdot + (1 - gamma) * np.cos(theta)*thetadot + gamma3 * np.cos(theta3r)*theta3rdot + (l3 - gamma3) * np.cos(theta3r)*theta3rdot + l4*np.cos(theta4r)*theta4rdot
x4ldot = xdot + (1 - gamma) * np.cos(theta)*thetadot + gamma3 * np.cos(theta3l)*theta3ldot + (l3 - gamma3) * np.cos(theta3l)*theta3ldot + l4*np.cos(theta4l)*theta4ldot

y4r = y - (l - gamma) * np.cos(theta) - gamma3 * np.cos(theta3r) - (l3 - gamma3) * np.cos(theta3r) - gamma4 * np.cos(theta4r)
y4l = y - (l - gamma) * np.cos(theta) - gamma3 * np.cos(theta3l) - (l3 - gamma3) * np.cos(theta3l) - gamma4 * np.cos(theta4l)

y4rdot = ydot + (1 - gamma) * np.sin(theta)*thetadot + gamma3*np.sin(theta3r)*theta3rdot + (l3 - gamma3)*np.sin(theta3r)*theta3rdot + gamma4*np.sin(theta4r)*theta4rdot
y4ldot = ydot + (1 - gamma) * np.sin(theta)*thetadot + gamma3*np.sin(theta3l)*theta3ldot + (l3 - gamma3)*np.sin(theta3l)*theta3ldot + gamma4*np.sin(theta4l)*theta4ldot

# Calculating foot positions and velocities
xfrdot = x4rdot + (l4 - gamma4)*theta4rdot*np.cos(theta4r)
xfldot = x4ldot - (l4 - gamma4)*theta4ldot*np.cos(theta4l)

yfr = y4r - (l4 - gamma4)*np.cos(theta4r)
yfl = y4l - (l4 - gamma4)*np.cos(theta4l)

yfrdot = y4rdot + (l4 - gamma4)*theta4rdot*np.sin(theta4r)
yfldot = y4ldot + (l4 - gamma4)*theta4ldot*np.sin(theta4l)

# Calculating floor reaction forces
fNr = np.maximum(0,-FLOOR_SPRING_STIFFNESS*yfr - (FLOOR_DAMPING_COEFFICIENT*yfrdot if yfr <= 0 else 0))

fNr = - (FLOOR_SPRING_STIFFNESS*yfr if yfr <= 0 else 0 + FLOOR_DAMPING_COEFFICIENT*yfrdot if yfr <= 0 and yfrdot <= 0 else 0)

fNl = np.maximum(0,-FLOOR_SPRING_STIFFNESS*yfl - (FLOOR_DAMPING_COEFFICIENT*yfldot if yfl <= 0 else 0))
fFr = (-FLOOR_MU*fNr*2/np.pi*np.arctan(xfrdot*ATANGAIN/ATANDEL))
fFl = (-FLOOR_MU*fNl*2/np.pi*np.arctan(xfldot*ATANGAIN/ATANDEL))

C = np.matrix([[	fFl + fFr - 2*gamma*m1*thetadot**2*np.sin(theta) - 2*gamma*m2*thetadot**2*np.sin(theta) - 2*gamma*m3*thetadot**2*np.sin(theta) - 2*gamma*m4*thetadot**2*np.sin(theta) + gamma1*m1*theta1ldot**2*np.sin(theta1l) + gamma1*m1*theta1rdot**2*np.sin(theta1r) + gamma2*m2*theta2ldot**2*np.sin(theta2l) + gamma2*m2*theta2rdot**2*np.sin(theta2r) + gamma3*m3*theta3ldot**2*np.sin(theta3l) + gamma3*m3*theta3rdot**2*np.sin(theta3r) + gamma4*m4*theta4ldot**2*np.sin(theta4l) + gamma4*m4*theta4rdot**2*np.sin(theta4r) + 2*l*m3*thetadot**2*np.sin(theta) + 2*l*m4*thetadot**2*np.sin(theta) + l1*m2*theta1ldot**2*np.sin(theta1l) + l1*m2*theta1rdot**2*np.sin(theta1r) + l3*m4*theta3ldot**2*np.sin(theta3l) + l3*m4*theta3rdot**2*np.sin(theta3r)	],
                [	fNl + fNr - g*m - 2*g*m1 - 2*g*m2 - 2*g*m3 - 2*g*m4 + 2*gamma*m1*thetadot**2*np.cos(theta) + 2*gamma*m2*thetadot**2*np.cos(theta) + 2*gamma*m3*thetadot**2*np.cos(theta) + 2*gamma*m4*thetadot**2*np.cos(theta) - gamma1*m1*theta1ldot**2*np.cos(theta1l) - gamma1*m1*theta1rdot**2*np.cos(theta1r) - gamma2*m2*theta2ldot**2*np.cos(theta2l) - gamma2*m2*theta2rdot**2*np.cos(theta2r) - gamma3*m3*theta3ldot**2*np.cos(theta3l) - gamma3*m3*theta3rdot**2*np.cos(theta3r) - gamma4*m4*theta4ldot**2*np.cos(theta4l) - gamma4*m4*theta4rdot**2*np.cos(theta4r) - 2*l*m3*thetadot**2*np.cos(theta) - 2*l*m4*thetadot**2*np.cos(theta) - l1*m2*theta1ldot**2*np.cos(theta1l) - l1*m2*theta1rdot**2*np.cos(theta1r) - l3*m4*theta3ldot**2*np.cos(theta3l) - l3*m4*theta3rdot**2*np.cos(theta3r)	],
                [	c1*theta1ldot + c1*theta1rdot - 2*c1*thetadot + c3*theta3ldot + c3*theta3rdot - 2*c3*thetadot - fFl*gamma*np.cos(theta) + fFl*l*np.cos(theta) - fFr*gamma*np.cos(theta) + fFr*l*np.cos(theta) - fNl*gamma*np.sin(theta) + fNl*l*np.sin(theta) - fNr*gamma*np.sin(theta) + fNr*l*np.sin(theta) + 2*g*gamma*m1*np.sin(theta) + 2*g*gamma*m2*np.sin(theta) + 2*g*gamma*m3*np.sin(theta) + 2*g*gamma*m4*np.sin(theta) - 2*g*l*m3*np.sin(theta) - 2*g*l*m4*np.sin(theta) + gamma*gamma1*m1*theta1ldot**2*np.sin(theta)*np.cos(theta1l) - gamma*gamma1*m1*theta1ldot**2*np.sin(theta1l)*np.cos(theta) + gamma*gamma1*m1*theta1rdot**2*np.sin(theta)*np.cos(theta1r) - gamma*gamma1*m1*theta1rdot**2*np.sin(theta1r)*np.cos(theta) + gamma*gamma2*m2*theta2ldot**2*np.sin(theta)*np.cos(theta2l) - gamma*gamma2*m2*theta2ldot**2*np.sin(theta2l)*np.cos(theta) + gamma*gamma2*m2*theta2rdot**2*np.sin(theta)*np.cos(theta2r) - gamma*gamma2*m2*theta2rdot**2*np.sin(theta2r)*np.cos(theta) + gamma*gamma3*m3*theta3ldot**2*np.sin(theta)*np.cos(theta3l) - gamma*gamma3*m3*theta3ldot**2*np.sin(theta3l)*np.cos(theta) + gamma*gamma3*m3*theta3rdot**2*np.sin(theta)*np.cos(theta3r) - gamma*gamma3*m3*theta3rdot**2*np.sin(theta3r)*np.cos(theta) + gamma*gamma4*m4*theta4ldot**2*np.sin(theta)*np.cos(theta4l) - gamma*gamma4*m4*theta4ldot**2*np.sin(theta4l)*np.cos(theta) + gamma*gamma4*m4*theta4rdot**2*np.sin(theta)*np.cos(theta4r) - gamma*gamma4*m4*theta4rdot**2*np.sin(theta4r)*np.cos(theta) + gamma*l1*m2*theta1ldot**2*np.sin(theta)*np.cos(theta1l) - gamma*l1*m2*theta1ldot**2*np.sin(theta1l)*np.cos(theta) + gamma*l1*m2*theta1rdot**2*np.sin(theta)*np.cos(theta1r) - gamma*l1*m2*theta1rdot**2*np.sin(theta1r)*np.cos(theta) + gamma*l3*m4*theta3ldot**2*np.sin(theta)*np.cos(theta3l) - gamma*l3*m4*theta3ldot**2*np.sin(theta3l)*np.cos(theta) + gamma*l3*m4*theta3rdot**2*np.sin(theta)*np.cos(theta3r) - gamma*l3*m4*theta3rdot**2*np.sin(theta3r)*np.cos(theta) - gamma3*l*m3*theta3ldot**2*np.sin(theta)*np.cos(theta3l) + gamma3*l*m3*theta3ldot**2*np.sin(theta3l)*np.cos(theta) - gamma3*l*m3*theta3rdot**2*np.sin(theta)*np.cos(theta3r) + gamma3*l*m3*theta3rdot**2*np.sin(theta3r)*np.cos(theta) - gamma4*l*m4*theta4ldot**2*np.sin(theta)*np.cos(theta4l) + gamma4*l*m4*theta4ldot**2*np.sin(theta4l)*np.cos(theta) - gamma4*l*m4*theta4rdot**2*np.sin(theta)*np.cos(theta4r) + gamma4*l*m4*theta4rdot**2*np.sin(theta4r)*np.cos(theta) - k1*phi1l - k1*phi1r - 2*k1*theta + k1*theta1l + k1*theta1r - k3*phi3l - k3*phi3r - 2*k3*theta + k3*theta3l + k3*theta3r - l*l3*m4*theta3ldot**2*np.sin(theta)*np.cos(theta3l) + l*l3*m4*theta3ldot**2*np.sin(theta3l)*np.cos(theta) - l*l3*m4*theta3rdot**2*np.sin(theta)*np.cos(theta3r) + l*l3*m4*theta3rdot**2*np.sin(theta3r)*np.cos(theta)	],
                [	-c1*theta1rdot + c1*thetadot - c2*theta1rdot + c2*theta2rdot - g*gamma1*m1*np.sin(theta1r) - g*l1*m2*np.sin(theta1r) - gamma*gamma1*m1*thetadot**2*np.sin(theta)*np.cos(theta1r) + gamma*gamma1*m1*thetadot**2*np.sin(theta1r)*np.cos(theta) - gamma*l1*m2*thetadot**2*np.sin(theta)*np.cos(theta1r) + gamma*l1*m2*thetadot**2*np.sin(theta1r)*np.cos(theta) - gamma2*l1*m2*theta2rdot**2*np.sin(theta1r)*np.cos(theta2r) + gamma2*l1*m2*theta2rdot**2*np.sin(theta2r)*np.cos(theta1r) + k1*phi1r + k1*theta - k1*theta1r - k2*phi2r - k2*theta1r + k2*theta2r	],
                [	c2*theta1rdot - c2*theta2rdot - g*gamma2*m2*np.sin(theta2r) - gamma*gamma2*m2*thetadot**2*np.sin(theta)*np.cos(theta2r) + gamma*gamma2*m2*thetadot**2*np.sin(theta2r)*np.cos(theta) + gamma2*l1*m2*theta1rdot**2*np.sin(theta1r)*np.cos(theta2r) - gamma2*l1*m2*theta1rdot**2*np.sin(theta2r)*np.cos(theta1r) + k2*phi2r + k2*theta1r - k2*theta2r	],
                [	-c3*theta3rdot + c3*thetadot - c4*theta3rdot + c4*theta4rdot + fFr*l3*np.cos(theta3r) + fNr*l3*np.sin(theta3r) - g*gamma3*m3*np.sin(theta3r) - g*l3*m4*np.sin(theta3r) - gamma*gamma3*m3*thetadot**2*np.sin(theta)*np.cos(theta3r) + gamma*gamma3*m3*thetadot**2*np.sin(theta3r)*np.cos(theta) - gamma*l3*m4*thetadot**2*np.sin(theta)*np.cos(theta3r) + gamma*l3*m4*thetadot**2*np.sin(theta3r)*np.cos(theta) + gamma3*l*m3*thetadot**2*np.sin(theta)*np.cos(theta3r) - gamma3*l*m3*thetadot**2*np.sin(theta3r)*np.cos(theta) - gamma4*l3*m4*theta4rdot**2*np.sin(theta3r)*np.cos(theta4r) + gamma4*l3*m4*theta4rdot**2*np.sin(theta4r)*np.cos(theta3r) + k3*phi3r + k3*theta - k3*theta3r - k4*phi4r - k4*theta3r + k4*theta4r + l*l3*m4*thetadot**2*np.sin(theta)*np.cos(theta3r) - l*l3*m4*thetadot**2*np.sin(theta3r)*np.cos(theta)	],
                [	c4*theta3rdot - c4*theta4rdot + fFr*l4*np.cos(theta4r) + fNr*l4*np.sin(theta4r) - g*gamma4*m4*np.sin(theta4r) - gamma*gamma4*m4*thetadot**2*np.sin(theta)*np.cos(theta4r) + gamma*gamma4*m4*thetadot**2*np.sin(theta4r)*np.cos(theta) + gamma4*l*m4*thetadot**2*np.sin(theta)*np.cos(theta4r) - gamma4*l*m4*thetadot**2*np.sin(theta4r)*np.cos(theta) + gamma4*l3*m4*theta3rdot**2*np.sin(theta3r)*np.cos(theta4r) - gamma4*l3*m4*theta3rdot**2*np.sin(theta4r)*np.cos(theta3r) + k4*phi4r + k4*theta3r - k4*theta4r	],
                [	-c1*theta1ldot + c1*thetadot - c2*theta1ldot + c2*theta2ldot - g*gamma1*m1*np.sin(theta1l) - g*l1*m2*np.sin(theta1l) - gamma*gamma1*m1*thetadot**2*np.sin(theta)*np.cos(theta1l) + gamma*gamma1*m1*thetadot**2*np.sin(theta1l)*np.cos(theta) - gamma*l1*m2*thetadot**2*np.sin(theta)*np.cos(theta1l) + gamma*l1*m2*thetadot**2*np.sin(theta1l)*np.cos(theta) - gamma2*l1*m2*theta2ldot**2*np.sin(theta1l)*np.cos(theta2l) + gamma2*l1*m2*theta2ldot**2*np.sin(theta2l)*np.cos(theta1l) + k1*phi1l + k1*theta - k1*theta1l - k2*phi2l - k2*theta1l + k2*theta2l	],
                [	c2*theta1ldot - c2*theta2ldot - g*gamma2*m2*np.sin(theta2l) - gamma*gamma2*m2*thetadot**2*np.sin(theta)*np.cos(theta2l) + gamma*gamma2*m2*thetadot**2*np.sin(theta2l)*np.cos(theta) + gamma2*l1*m2*theta1ldot**2*np.sin(theta1l)*np.cos(theta2l) - gamma2*l1*m2*theta1ldot**2*np.sin(theta2l)*np.cos(theta1l) + k2*phi2l + k2*theta1l - k2*theta2l	],
                [	-c3*theta3ldot + c3*thetadot - c4*theta3ldot + c4*theta4ldot + fFl*l3*np.cos(theta3l) + fNl*l3*np.sin(theta3l) - g*gamma3*m3*np.sin(theta3l) - g*l3*m4*np.sin(theta3l) - gamma*gamma3*m3*thetadot**2*np.sin(theta)*np.cos(theta3l) + gamma*gamma3*m3*thetadot**2*np.sin(theta3l)*np.cos(theta) - gamma*l3*m4*thetadot**2*np.sin(theta)*np.cos(theta3l) + gamma*l3*m4*thetadot**2*np.sin(theta3l)*np.cos(theta) + gamma3*l*m3*thetadot**2*np.sin(theta)*np.cos(theta3l) - gamma3*l*m3*thetadot**2*np.sin(theta3l)*np.cos(theta) - gamma4*l3*m4*theta4ldot**2*np.sin(theta3l)*np.cos(theta4l) + gamma4*l3*m4*theta4ldot**2*np.sin(theta4l)*np.cos(theta3l) + k3*phi3l + k3*theta - k3*theta3l - k4*phi4l - k4*theta3l + k4*theta4l + l*l3*m4*thetadot**2*np.sin(theta)*np.cos(theta3l) - l*l3*m4*thetadot**2*np.sin(theta3l)*np.cos(theta)	],
                [	c4*theta3ldot - c4*theta4ldot + fFl*l4*np.cos(theta4l) + fNl*l4*np.sin(theta4l) - g*gamma4*m4*np.sin(theta4l) - gamma*gamma4*m4*thetadot**2*np.sin(theta)*np.cos(theta4l) + gamma*gamma4*m4*thetadot**2*np.sin(theta4l)*np.cos(theta) + gamma4*l*m4*thetadot**2*np.sin(theta)*np.cos(theta4l) - gamma4*l*m4*thetadot**2*np.sin(theta4l)*np.cos(theta) + gamma4*l3*m4*theta3ldot**2*np.sin(theta3l)*np.cos(theta4l) - gamma4*l3*m4*theta3ldot**2*np.sin(theta4l)*np.cos(theta3l) + k4*phi4l + k4*theta3l - k4*theta4l	]])


# Calculating second derivatives
second_derivatives = np.array(np.linalg.inv(M)*(C)).squeeze()

print(second_derivatives)
