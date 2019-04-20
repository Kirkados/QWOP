# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 08:36:24 2019

@author: Kirk
"""
import sympy as sp

# State symbols
x,       y,    theta, \
x1r,    y1r,    theta1r,  \
x2r,    y2r,    theta2r,   \
x3r,    y3r,    theta3r,  \
x4r,    y4r,    theta4r,  \
x1l,    y1l,    theta1l,  \
x2l,    y2l,    theta2l,  \
x3l,    y3l,    theta3l,  \
x4l,    y4l,    theta4l = sp.symbols('x,       y,    theta,    x1r,    y1r,    theta1r,    x2r,    y2r,    theta2r,    x3r,    y3r,    theta3r,    x4r,    y4r,    theta4r,    x1l,    y1l,    theta1l,    x2l,    y2l,    theta2l,    x3l,    y3l,    theta3l,    x4l,    y4l,    theta4l')

# State derivative symbols
xdot,       ydot,    thetadot, \
x1rdot,    y1rdot,    theta1rdot,  \
x2rdot,    y2rdot,    theta2rdot,   \
x3rdot,    y3rdot,    theta3rdot,  \
x4rdot,    y4rdot,    theta4rdot,  \
x1ldot,    y1ldot,    theta1ldot,  \
x2ldot,    y2ldot,    theta2ldot,  \
x3ldot,    y3ldot,    theta3ldot,  \
x4ldot,    y4ldot,    theta4ldot = sp.symbols('xdot,       ydot,    thetadot,    x1rdot,    y1rdot,    theta1rdot,    x2rdot,    y2rdot,    theta2rdot,    x3rdot,    y3rdot,    theta3rdot,    x4rdot,    y4rdot,    theta4rdot,    x1ldot,    y1ldot,    theta1ldot,    x2ldot,    y2dot,    theta2ldot,    x3ldot,    y3ldot,    theta3ldot,    x4ldot,    y4ldot,    theta4ldot')

# State second derivative symbols
xdotdot,       ydotdot,    thetadotdot, \
x1rdotdot,    y1rdotdot,    theta1rdotdot,  \
x2rdotdot,    y2rdotdot,    theta2rdotdot,   \
x3rdotdot,    y3rdotdot,    theta3rdotdot,  \
x4rdotdot,    y4rdotdot,    theta4rdotdot,  \
x1ldotdot,    y1ldotdot,    theta1ldotdot,  \
x2ldotdot,    y2ldotdot,    theta2ldotdot,  \
x3ldotdot,    y3ldotdot,    theta3ldotdot,  \
x4ldotdot,    y4ldotdot,    theta4ldotdot = sp.symbols('xdotdot,       ydotdot,    thetadotdot,    x1rdotdot,    y1rdotdot,    theta1rdotdot,    x2rdotdot,    y2rdotdot,    theta2rdotdot,    x3rdotdot,    y3rdotdot,    theta3rdotdot,    x4rdotdot,    y4rdotdot,    theta4rdotdot,    x1ldotdot,    y1ldotdot,    theta1ldotdot,    x2ldotdot,    y2ldotdot,    theta2ldotdot,    x3ldotdot,    y3ldotdot,    theta3ldotdot,    x4ldotdot,    y4ldotdot,    theta4ldotdot')

# Parameter Symbols
m, m1, m2, m3, m4, \
l, l1, l2, l3, l4, \
gamma, gamma1, gamma2, gamma3, gamma4,\
I, I1, I2, I3, I4, \
k3, c3, k4, c4, \
k1, c1, k2, c2, \
g, FLOOR_SPRING_STIFFNESS, FLOOR_DAMPING_COEFFICIENT, FLOOR_FRICTION_STIFFNESS, FLOOR_MU, ATANDEL, ATANGAIN, \
phi1r, phi2r, phi3r, phi4r, phi1l, phi2l, phi3l, phi4l = sp.symbols(  'm, m1, m2, m3, m4, \
                                                                    l, l1, l2, l3, l4, \
                                                                    gamma, gamma1, gamma2, gamma3, gamma4,\
                                                                    I, I1, I2, I3, I4, \
                                                                    k3, c3, k4, c4, \
                                                                    k1, c1, k2, c2, \
                                                                    g, FLOOR_SPRING_STIFFNESS, FLOOR_DAMPING_COEFFICIENT, FLOOR_FRICTION_STIFFNESS, FLOOR_MU, ATANDEL, ATANGAIN, \
                                                                    phi1r, phi2r, phi3r, phi4r, phi1l, phi2l, phi3l, phi4l') 

# Friction symbols
fFr, fFl, fNr, fNl = sp.symbols('fFr, fFl, fNr, fNl')


M = sp.Matrix([[	m	,	0	,	0	,	m1	,	0	,	0	,	m2	,	0	,	0	,	m3	,	0	,	0	,	m4	,	0	,	0	,	m1	,	0	,	0	,	m2	,	0	,	0	,	m3	,	0	,	0	,	m4	,	0	,	0	],
                    [	0	,	m	,	0	,	0	,	m1	,	0	,	0	,	m2	,	0	,	0	,	m3	,	0	,	0	,	m4	,	0	,	0	,	m1	,	0	,	0	,	m2	,	0	,	0	,	m3	,	0	,	0	,	m4	,	0	],
                    [	0	,	0	,	I	,	-m1*gamma*sp.cos(theta)	,	-m1*gamma*sp.sin(theta)	,	0	,	-m2*gamma*sp.cos(theta)	,	-m2*gamma*sp.sin(theta)	,	0	,	m3*(l-gamma)*sp.cos(theta)	,	m3*(l-gamma)*sp.sin(theta)	,	0	,	m4*(l-gamma)*sp.cos(theta)	,	m4*(l-gamma)*sp.sin(theta)	,	0	,	-m1*gamma*sp.cos(theta)	,	-m1*gamma*sp.sin(theta)	,	0	,	-m2*gamma*sp.cos(theta)	,	-m2*gamma*sp.sin(theta)	,	0	,	m3*(l-gamma)*sp.cos(theta)	,	m3*(l-gamma)*sp.sin(theta)	,	0	,	m4*(l-gamma)*sp.cos(theta)	,	m4*(l-gamma)*sp.sin(theta)	,	0	],
                    [	0	,	0	,	0	,	m1*gamma1*sp.cos(theta1r)	,	m1*gamma1*sp.sin(theta1r)	,	I1	,	m2*(l1)*sp.cos(theta1r)	,	m2*(l1)*sp.sin(theta1r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	m2*gamma2*sp.cos(theta2r)	,	m2*gamma2*sp.sin(theta2r)	,	I2	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m3*gamma3*sp.cos(theta3r)	,	m3*gamma3*sp.sin(theta3r)	,	I3	,	m4*(l3)*sp.cos(theta3r)	,	m4*(l3)*sp.sin(theta3r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m4*gamma4*sp.cos(theta4r)	,	m4*gamma4*sp.sin(theta4r)	,	I4	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	-1	,	0	,	gamma*sp.cos(theta)	,	1	,	0	,	-gamma1*sp.cos(theta1r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	-1	,	gamma*sp.sin(theta)	,	0	,	1	,	-gamma1*sp.sin(theta1r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	-1	,	0	,	-(l1-gamma1)*sp.cos(theta1r)	,	1	,	0	,	-gamma2*sp.cos(theta2r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	-1	,	-(l1-gamma1)*sp.sin(theta1r)	,	0	,	1	,	-gamma2*sp.sin(theta2r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	-1	,	0	,	-(l-gamma)*sp.cos(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	0	,	-gamma3*sp.cos(theta3r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	-1	,	-(l-gamma)*sp.sin(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	-gamma3*sp.sin(theta3r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	0	,	-(l3-gamma3)*sp.cos(theta3r)	,	1	,	0	,	-gamma4*sp.cos(theta4r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	-(l3-gamma3)*sp.sin(theta3r)	,	0	,	1	,	-gamma4*sp.sin(theta4r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m1*gamma1*sp.cos(theta1l)	,	m1*gamma1*sp.sin(theta1l)	,	I1	,	m2*(l1)*sp.cos(theta1l)	,	m2*(l1)*sp.sin(theta1l)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m2*gamma2*sp.cos(theta2l)	,	m2*gamma2*sp.sin(theta2l)	,	I2	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m3*gamma3*sp.cos(theta3l)	,	m3*gamma3*sp.sin(theta3l)	,	I3	,	m4*(l3)*sp.cos(theta3l)	,	m4*(l3)*sp.sin(theta3l)	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m4*gamma4*sp.cos(theta4l)	,	m4*gamma4*sp.sin(theta4l)	,	I4	],
                    [	-1	,	0	,	gamma*sp.cos(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	0	,	-gamma1*sp.cos(theta1l)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	-1	,	gamma*sp.sin(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	-gamma1*sp.sin(theta1l)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	0	,	-(l1-gamma1)*sp.cos(theta1l)	,	1	,	0	,	-gamma2*sp.cos(theta2l)	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	-(l1-gamma1)*sp.sin(theta1l)	,	0	,	1	,	-gamma2*sp.sin(theta2l)	,	0	,	0	,	0	,	0	,	0	,	0	],
                    [	-1	,	0	,	-(l-gamma)*sp.cos(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	0	,	-gamma3*sp.cos(theta3l)	,	0	,	0	,	0	],
                    [	0	,	-1	,	-(l-gamma)*sp.sin(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	-gamma3*sp.sin(theta3l)	,	0	,	0	,	0	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	0	,	-(l3-gamma3)*sp.cos(theta3l)	,	1	,	0	,	-gamma4*sp.cos(theta4l)	],
                    [	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	-(l3-gamma3)*sp.sin(theta3l)	,	0	,	1	,	-gamma4*sp.sin(theta4l)	]])
               
state = sp.Matrix([[xdotdot],
                  [ydotdot],
                  [thetadotdot],
                  [x1rdotdot],
                  [y1rdotdot],
                  [theta1rdotdot],
                  [x2rdotdot],
                  [y2rdotdot],
                  [theta2rdotdot],
                  [x3rdotdot],
                  [y3rdotdot],
                  [theta3rdotdot],
                  [x4rdotdot],
                  [y4rdotdot],
                  [theta4rdotdot],
                  [x1ldotdot],
                  [y1ldotdot],
                  [theta1ldotdot],
                  [x2ldotdot],
                  [y2ldotdot],
                  [theta2ldotdot],
                  [x3ldotdot],
                  [y3ldotdot],
                  [theta3ldotdot],
                  [x4ldotdot],
                  [y4ldotdot],
                  [theta4ldotdot]])

C = sp.Matrix([[	fFr+fFl	],
                    [	-m*g-2*m1*g-2*m2*g-2*m3*g-2*m4*g+fNr+fNl	],
                    [	2*m1*g*gamma*sp.sin(theta)+2*m2*g*gamma*sp.sin(theta)+fFr*(l-gamma)*sp.cos(theta)+fFl*(l-gamma)*sp.cos(theta)-2*m3*g*(l-gamma)*sp.sin(theta)-2*m4*g*(l-gamma)*sp.sin(theta)+fNr*(l-gamma)*sp.sin(theta)+fNl*(l-gamma)*sp.sin(theta)-k1*(theta+phi1l-theta1l)-c1*(thetadot-theta1ldot)-k1*(theta+phi1r-theta1r)-c1*(thetadot-theta1rdot)-k3*(theta+phi3l-theta3l)-c3*(thetadot-theta3ldot)-k3*(theta+phi3r-theta3r)-c3*(thetadot-theta3rdot)	],
                    [	-m1*g*gamma1*sp.sin(theta1r)-m2*g*(l1)*sp.sin(theta1r)-k2*(theta1r+phi2r-theta2r)-c2*(theta1rdot-theta2rdot)+k1*(theta+phi1r-theta1r)+c1*(thetadot-theta1rdot)	],
                    [	-m2*g*gamma2*sp.sin(theta2r)+k2*(theta1r+phi2r-theta2r)+c2*(theta1rdot-theta2rdot)	],
                    [	-m3*g*gamma3*sp.sin(theta3r)-m4*g*(l3)*sp.sin(theta3r)-k4*(theta3r+phi4r-theta4r)-c4*(theta3rdot-theta4rdot)+k3*(theta+phi3r-theta3r)+c3*(thetadot-theta3rdot)+fFr*(l3)*sp.cos(theta3r)+fNr*(l3)*sp.sin(theta3r)	],
                    [	-m4*g*gamma4*sp.sin(theta4r)+k4*(theta3r+phi4r-theta4r)+c4*(theta3rdot-theta4rdot)+fFr*(l4)*sp.cos(theta4r)+fNr*(l4)*sp.sin(theta4r)	],
                    [	gamma*thetadot**2*sp.sin(theta)-gamma1*theta1rdot**2*sp.sin(theta1r)	],
                    [	-gamma*thetadot**2*sp.cos(theta)+gamma1*theta1rdot**2*sp.cos(theta1r)	],
                    [	-(l1-gamma1)*theta1rdot**2*sp.sin(theta1r)-gamma2*theta2rdot**2*sp.sin(theta2r)	],
                    [	(l1-gamma1)*theta1rdot**2*sp.cos(theta1r)+gamma2*theta2rdot**2*sp.cos(theta2r)	],
                    [	-(l-gamma)*thetadot**2*sp.sin(theta)-gamma3*theta3rdot**2*sp.sin(theta3r)	],
                    [	(l-gamma)*thetadot**2*sp.cos(theta)+gamma3*theta3rdot**2*sp.cos(theta3r)	],
                    [	-(l3-gamma3)*theta3rdot**2*sp.sin(theta3r)-gamma4*theta4rdot**2*sp.sin(theta4r)	],
                    [	(l3-gamma3)*theta3rdot**2*sp.cos(theta3r)+gamma4*theta4rdot**2*sp.cos(theta4r)	],
                    [	-m1*g*gamma1*sp.sin(theta1l)-m2*g*(l1)*sp.sin(theta1l)-k2*(theta1l+phi2l-theta2l)-c2*(theta1ldot-theta2ldot)+k1*(theta+phi1l-theta1l)+c1*(thetadot-theta1ldot)	],
                    [	-m2*g*gamma2*sp.sin(theta2l)+k2*(theta1l+phi2l-theta2l)+c2*(theta1ldot-theta2ldot)	],
                    [	-m3*g*gamma3*sp.sin(theta3l)-m4*g*(l3)*sp.sin(theta3l)-k4*(theta3l+phi4l-theta4l)-c4*(theta3ldot-theta4ldot)+k3*(theta+phi3l-theta3l)+c3*(thetadot-theta3ldot)+fFl*(l3)*sp.cos(theta3l)+fNl*(l3)*sp.sin(theta3l)	],
                    [	-m4*g*gamma4*sp.sin(theta4l)+k4*(theta3l+phi4l-theta4l)+c4*(theta3ldot-theta4ldot)+fFl*(l4)*sp.cos(theta4l)+fNl*(l4)*sp.sin(theta4l)	],
                    [	gamma*thetadot**2*sp.sin(theta)-gamma1*theta1ldot**2*sp.sin(theta1l)	],
                    [	-gamma*thetadot**2*sp.cos(theta)+gamma1*theta1ldot**2*sp.cos(theta1l)	],
                    [	-(l1-gamma1)*theta1ldot**2*sp.sin(theta1l)-gamma2*theta2ldot**2*sp.sin(theta2l)	],
                    [	(l1-gamma1)*theta1ldot**2*sp.cos(theta1l)+gamma2*theta2ldot**2*sp.cos(theta2l)	],
                    [	-(l-gamma)*thetadot**2*sp.sin(theta)-gamma3*theta3ldot**2*sp.sin(theta3l)	],
                    [	(l-gamma)*thetadot**2*sp.cos(theta)+gamma3*theta3ldot**2*sp.cos(theta3l)	],
                    [	-(l3-gamma3)*theta3ldot**2*sp.sin(theta3l)-gamma4*theta4ldot**2*sp.sin(theta4l)	],
                    [	(l3-gamma3)*theta3ldot**2*sp.cos(theta3l)+gamma4*theta4ldot**2*sp.cos(theta4l)	]])
     


Eqs = M*state - C


e1 = Eqs[0:7] 
e2 = Eqs[15:19]
e3=e1+e2
#print(e3)

fr = Eqs[7:15]
fl = Eqs[19:]

#Eqs_subs = 

#solution = sp.nonlinsolve(Eqs, [xdotdot, ydotdot, thetadotdot, theta1rdotdot, theta2rdotdot, theta3rdotdot, theta4rdotdot, theta1ldotdot, theta2ldotdot, theta3ldotdot, theta4ldotdot])
#solve for X and Y (right)

eq_x1rdotdot = sp.solve(fr[0],x1rdotdot)[0]
eq_y1rdotdot = sp.solve(fr[1],y1rdotdot)[0]
eq_x2rdotdot = sp.solve(fr[2].subs(x1rdotdot, eq_x1rdotdot),x2rdotdot)[0]
eq_y2rdotdot = sp.solve(fr[3].subs(y1rdotdot, eq_y1rdotdot),y2rdotdot)[0]
eq_x3rdotdot = sp.solve(fr[4],x3rdotdot)[0]
eq_y3rdotdot = sp.solve(fr[5],y3rdotdot)[0]
eq_x4rdotdot = sp.solve(fr[6].subs(x3rdotdot, eq_x3rdotdot),x4rdotdot)[0]
eq_y4rdotdot = sp.solve(fr[7].subs(y3rdotdot, eq_y3rdotdot),y4rdotdot)[0]

eq_x1ldotdot = sp.solve(fl[0],x1ldotdot)[0]
eq_y1ldotdot = sp.solve(fl[1],y1ldotdot)[0]
eq_x2ldotdot = sp.solve(fl[2].subs(x1ldotdot, eq_x1ldotdot),x2ldotdot)[0]
eq_y2ldotdot = sp.solve(fl[3].subs(y1ldotdot, eq_y1ldotdot),y2ldotdot)[0]
eq_x3ldotdot = sp.solve(fl[4],x3ldotdot)[0]
eq_y3ldotdot = sp.solve(fl[5],y3ldotdot)[0]
eq_x4ldotdot = sp.solve(fl[6].subs(x3ldotdot, eq_x3ldotdot),x4ldotdot)[0]
eq_y4ldotdot = sp.solve(fl[7].subs(y3ldotdot, eq_y3ldotdot),y4ldotdot)[0]


#eq1=Eqs[0].subs([(x1rdotdot, eq_x1rdotdot),(x2rdotdot, eq_x2rdotdot),(x3rdotdot, eq_x3rdotdot),(x4rdotdot, eq_x4rdotdot),(x1ldotdot, eq_x1ldotdot),(x2ldotdot, eq_x2ldotdot),(x3ldotdot, eq_x3ldotdot),(x4ldotdot, eq_x4ldotdot)])
#eq2=Eqs[1].subs([(y1rdotdot, eq_y1rdotdot),(y2rdotdot, eq_y2rdotdot),(y3rdotdot, eq_y3rdotdot),(y4rdotdot, eq_y4rdotdot),(y1ldotdot, eq_y1ldotdot),(y2ldotdot, eq_y2ldotdot),(y3ldotdot, eq_y3ldotdot),(y4ldotdot, eq_y4ldotdot)])
eq3=Eqs[2].subs([(x1rdotdot, eq_x1rdotdot),(x2rdotdot, eq_x2rdotdot),(x3rdotdot, eq_x3rdotdot),(x4rdotdot, eq_x4rdotdot),(x1ldotdot, eq_x1ldotdot),(x2ldotdot, eq_x2ldotdot),(x3ldotdot, eq_x3ldotdot),(x4ldotdot, eq_x4ldotdot),(y1rdotdot, eq_y1rdotdot),(y2rdotdot, eq_y2rdotdot),(y3rdotdot, eq_y3rdotdot),(y4rdotdot, eq_y4rdotdot),(y1ldotdot, eq_y1ldotdot),(y2ldotdot, eq_y2ldotdot),(y3ldotdot, eq_y3ldotdot),(y4ldotdot, eq_y4ldotdot)])
#eq4=Eqs[3].subs([(x1rdotdot, eq_x1rdotdot),(x2rdotdot, eq_x2rdotdot),(y1rdotdot, eq_y1rdotdot),(y2rdotdot, eq_y2rdotdot)])
eq5=Eqs[4].subs([(x2rdotdot, eq_x2rdotdot),(y2rdotdot, eq_y2rdotdot)])
#eq6=Eqs[5].subs([(x3rdotdot, eq_x3rdotdot),(x4rdotdot, eq_x4rdotdot),(y3rdotdot, eq_y3rdotdot),(y4rdotdot, eq_y4rdotdot)])
#eq7=Eqs[6].subs([(x4rdotdot, eq_x4rdotdot),(y4rdotdot, eq_y4rdotdot)])
#eq8=Eqs[15].subs([(x1ldotdot, eq_x1ldotdot),(x2ldotdot, eq_x2ldotdot),(y1ldotdot, eq_y1ldotdot),(y2ldotdot, eq_y2ldotdot)])
#eq9=Eqs[16].subs([(x2ldotdot, eq_x2ldotdot),(y2ldotdot, eq_y2ldotdot)])
#eq10=Eqs[17].subs([(x3ldotdot, eq_x3ldotdot),(x4ldotdot, eq_x4ldotdot),(y3ldotdot, eq_y3ldotdot),(y4ldotdot, eq_y4ldotdot)])
#eq11=Eqs[18].subs([(x4ldotdot, eq_x4ldotdot),(y4ldotdot, eq_y4ldotdot)])

print(sp.collect(eq3, thetadotdot).coeff(thetadotdot,1))


#print(sp.simplify(eq3))
#Eqs_out = eq1+eq2+eq3+eq4+eq5+eq6+eq7+eq8+eq9+eq10+eq11












#print(eq_x1rdotdot)
#print(Eqs[0])
#e=Eqs[0].subs([(x1rdotdot, eq_x1rdotdot)])
#print(e)
#print(sp.solve(Eqs[8],y1rdotdot)
#print(sp.solve(Eqs[9],x2rdotdot)
#print(sp.solve(Eqs[10],y2rdotdot)
#print(sp.solve(Eqs[11],x3rdotdot)
#print(sp.solve(Eqs[12],y3rdotdot)
#print(sp.solve(Eqs[13],x4rdotdot)
#print(sp.solve(Eqs[14],y4rdotdot)
##solve for X and Y (left)
#print(sp.solve(Eqs[19],x1ldotdot)
#print(sp.solve(Eqs[20],y1ldotdot)
#print(sp.solve(Eqs[21],x2ldotdot)
#print(sp.solve(Eqs[22],y2ldotdot)
#print(sp.solve(Eqs[23],x3ldotdot)
#print(sp.solve(Eqs[24],y3ldotdot)
#print(sp.solve(Eqs[25],x4ldotdot)
#print(sp.solve(Eqs[26],y4ldotdot)




#print(sp.solve(Eqs[26],theta4ldotdot))