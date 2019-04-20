syms fFr fFl fNr fNl m m1 m2 m3 m4 ...
    l l1 l2 l3 l4...
    gamma gamma1 gamma2 gamma3 gamma4...
    I I1 I2 I3 I4...
    k3 c3 k4 c4...
    k1 c1 k2 c2...
    g FLOOR_SPRING_STIFFNESS FLOOR_DAMPING_COEFFICIENT FLOOR_FRICTION_STIFFNESS FLOOR_MU ATANDEL ATANGAIN...
    phi1r phi2r phi3r phi4r phi1l phi2l phi3l phi4l...
    x       y      theta...
    x1r   y1r    theta1r...
    x2r   y2r    theta2r...
    x3r   y3r    theta3r...
    x4r   y4r    theta4r...
    x1l   y1l    theta1l...
    x2l   y2l    theta2l...
    x3l   y3l    theta3l...
    x4l   y4l    theta4l...
    xdot     ydot      thetadot...
    x1rdot   y1rdot    theta1rdot...
    x2rdot   y2rdot    theta2rdot...
    x3rdot   y3rdot    theta3rdot...
    x4rdot   y4rdot    theta4rdot...
    x1ldot   y1ldot    theta1ldot...
    x2ldot   y2ldot    theta2ldot...
    x3ldot   y3ldot    theta3ldot...
    x4ldot   y4ldot    theta4ldot...
    xdotdot     ydotdot      thetadotdot...
    x1rdotdot   y1rdotdot    theta1rdotdot...
    x2rdotdot   y2rdotdot    theta2rdotdot...
    x3rdotdot   y3rdotdot    theta3rdotdot...
    x4rdotdot   y4rdotdot    theta4rdotdot...
    x1ldotdot   y1ldotdot    theta1ldotdot...
    x2ldotdot   y2ldotdot    theta2ldotdot...
    x3ldotdot   y3ldotdot    theta3ldotdot...
    x4ldotdot   y4ldotdot    theta4ldotdot



M = [	m	,	0	,	0	,	m1	,	0	,	0	,	m2	,	0	,	0	,	m3	,	0	,	0	,	m4	,	0	,	0	,	m1	,	0	,	0	,	m2	,	0	,	0	,	m3	,	0	,	0	,	m4	,	0	,	0	;
    0	,	m	,	0	,	0	,	m1	,	0	,	0	,	m2	,	0	,	0	,	m3	,	0	,	0	,	m4	,	0	,	0	,	m1	,	0	,	0	,	m2	,	0	,	0	,	m3	,	0	,	0	,	m4	,	0	;
    0	,	0	,	I	,	-m1*gamma*cos(theta)	,	-m1*gamma*sin(theta)	,	0	,	-m2*gamma*cos(theta)	,	-m2*gamma*sin(theta)	,	0	,	m3*(l-gamma)*cos(theta)	,	m3*(l-gamma)*sin(theta)	,	0	,	m4*(l-gamma)*cos(theta)	,	m4*(l-gamma)*sin(theta)	,	0	,	-m1*gamma*cos(theta)	,	-m1*gamma*sin(theta)	,	0	,	-m2*gamma*cos(theta)	,	-m2*gamma*sin(theta)	,	0	,	m3*(l-gamma)*cos(theta)	,	m3*(l-gamma)*sin(theta)	,	0	,	m4*(l-gamma)*cos(theta)	,	m4*(l-gamma)*sin(theta)	,	0	;
    0	,	0	,	0	,	m1*gamma1*cos(theta1r)	,	m1*gamma1*sin(theta1r)	,	I1	,	m2*(l1)*cos(theta1r)	,	m2*(l1)*sin(theta1r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	0	,	0	,	0	,	0	,	0	,	m2*gamma2*cos(theta2r)	,	m2*gamma2*sin(theta2r)	,	I2	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m3*gamma3*cos(theta3r)	,	m3*gamma3*sin(theta3r)	,	I3	,	m4*(l3)*cos(theta3r)	,	m4*(l3)*sin(theta3r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m4*gamma4*cos(theta4r)	,	m4*gamma4*sin(theta4r)	,	I4	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    -1	,	0	,	gamma*cos(theta)	,	1	,	0	,	-gamma1*cos(theta1r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	-1	,	gamma*sin(theta)	,	0	,	1	,	-gamma1*sin(theta1r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	0	,	0	,	-1	,	0	,	-(l1-gamma1)*cos(theta1r)	,	1	,	0	,	-gamma2*cos(theta2r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	0	,	0	,	0	,	-1	,	-(l1-gamma1)*sin(theta1r)	,	0	,	1	,	-gamma2*sin(theta2r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    -1	,	0	,	-(l-gamma)*cos(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	0	,	-gamma3*cos(theta3r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	-1	,	-(l-gamma)*sin(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	-gamma3*sin(theta3r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	0	,	-(l3-gamma3)*cos(theta3r)	,	1	,	0	,	-gamma4*cos(theta4r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	-(l3-gamma3)*sin(theta3r)	,	0	,	1	,	-gamma4*sin(theta4r)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m1*gamma1*cos(theta1l)	,	m1*gamma1*sin(theta1l)	,	I1	,	m2*(l1)*cos(theta1l)	,	m2*(l1)*sin(theta1l)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m2*gamma2*cos(theta2l)	,	m2*gamma2*sin(theta2l)	,	I2	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m3*gamma3*cos(theta3l)	,	m3*gamma3*sin(theta3l)	,	I3	,	m4*(l3)*cos(theta3l)	,	m4*(l3)*sin(theta3l)	,	0	;
    0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	m4*gamma4*cos(theta4l)	,	m4*gamma4*sin(theta4l)	,	I4	;
    -1	,	0	,	gamma*cos(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	0	,	-gamma1*cos(theta1l)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	-1	,	gamma*sin(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	-gamma1*sin(theta1l)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	0	,	-(l1-gamma1)*cos(theta1l)	,	1	,	0	,	-gamma2*cos(theta2l)	,	0	,	0	,	0	,	0	,	0	,	0	;
    0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	-(l1-gamma1)*sin(theta1l)	,	0	,	1	,	-gamma2*sin(theta2l)	,	0	,	0	,	0	,	0	,	0	,	0	;
    -1	,	0	,	-(l-gamma)*cos(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	0	,	-gamma3*cos(theta3l)	,	0	,	0	,	0	;
    0	,	-1	,	-(l-gamma)*sin(theta)	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	1	,	-gamma3*sin(theta3l)	,	0	,	0	,	0	;
    0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	0	,	-(l3-gamma3)*cos(theta3l)	,	1	,	0	,	-gamma4*cos(theta4l)	;
    0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	-1	,	-(l3-gamma3)*sin(theta3l)	,	0	,	1	,	-gamma4*sin(theta4l)];

state = [xdotdot       ;
    ydotdot       ;
    thetadotdot   ;
    x1rdotdot     ;
    y1rdotdot     ;
    theta1rdotdot ;
    x2rdotdot     ;
    y2rdotdot     ;
    theta2rdotdot ;
    x3rdotdot     ;
    y3rdotdot     ;
    theta3rdotdot ;
    x4rdotdot     ;
    y4rdotdot     ;
    theta4rdotdot ;
    x1ldotdot     ;
    y1ldotdot     ;
    theta1ldotdot ;
    x2ldotdot     ;
    y2ldotdot     ;
    theta2ldotdot ;
    x3ldotdot     ;
    y3ldotdot     ;
    theta3ldotdot ;
    x4ldotdot     ;
    y4ldotdot     ;
    theta4ldotdot];

C = [fFr+fFl                                                                                                                                                                                                                                                                                                                                                                                                                                ;
    -m*g-2*m1*g-2*m2*g-2*m3*g-2*m4*g+fNr+fNl                                                                                                                                                                                                                                                                                                                                                                                                    ;
    2*m1*g*gamma*sin(theta)+2*m2*g*gamma*sin(theta)+fFr*(l-gamma)*cos(theta)+fFl*(l-gamma)*cos(theta)-2*m3*g*(l-gamma)*sin(theta)-2*m4*g*(l-gamma)*sin(theta)+fNr*(l-gamma)*sin(theta)+fNl*(l-gamma)*sin(theta)-k1*(theta+phi1l-theta1l)-c1*(thetadot-theta1ldot)-k1*(theta+phi1r-theta1r)-c1*(thetadot-theta1rdot)-k3*(theta+phi3l-theta3l)-c3*(thetadot-theta3ldot)-k3*(theta+phi3r-theta3r)-c3*(thetadot-theta3rdot) ;
    -m1*g*gamma1*sin(theta1r)-m2*g*(l1)*sin(theta1r)-k2*(theta1r+phi2r-theta2r)-c2*(theta1rdot-theta2rdot)+k1*(theta+phi1r-theta1r)+c1*(thetadot-theta1rdot)                                                                                                                                                                                                                                                                              ;
    -m2*g*gamma2*sin(theta2r)+k2*(theta1r+phi2r-theta2r)+c2*(theta1rdot-theta2rdot)	                                                                                                                                                                                                                                                                                                                                                        ;
    -m3*g*gamma3*sin(theta3r)-m4*g*(l3)*sin(theta3r)-k4*(theta3r+phi4r-theta4r)-c4*(theta3rdot-theta4rdot)+k3*(theta+phi3r-theta3r)+c3*(thetadot-theta3rdot)+fFr*(l3)*cos(theta3r)+fNr*(l3)*sin(theta3r)                                                                                                                                                                                                                            ;
    -m4*g*gamma4*sin(theta4r)+k4*(theta3r+phi4r-theta4r)+c4*(theta3rdot-theta4rdot)+fFr*(l4)*cos(theta4r)+fNr*(l4)*sin(theta4r)	                                                                                                                                                                                                                                                                                                    ;
    gamma*thetadot^2*sin(theta)-gamma1*theta1rdot^2*sin(theta1r)	                                                                                                                                                                                                                                                                                                                                                                    ;
    -gamma*thetadot^2*cos(theta)+gamma1*theta1rdot^2*cos(theta1r)	                                                                                                                                                                                                                                                                                                                                                                    ;
    -(l1-gamma1)*theta1rdot^2*sin(theta1r)-gamma2*theta2rdot^2*sin(theta2r)	                                                                                                                                                                                                                                                                                                                                                            ;
    (l1-gamma1)*theta1rdot^2*cos(theta1r)+gamma2*theta2rdot^2*cos(theta2r)	                                                                                                                                                                                                                                                                                                                                                            ;
    -(l-gamma)*thetadot^2*sin(theta)-gamma3*theta3rdot^2*sin(theta3r)	                                                                                                                                                                                                                                                                                                                                                                ;
    (l-gamma)*thetadot^2*cos(theta)+gamma3*theta3rdot^2*cos(theta3r)	                                                                                                                                                                                                                                                                                                                                                                ;
    -(l3-gamma3)*theta3rdot^2*sin(theta3r)-gamma4*theta4rdot^2*sin(theta4r)	                                                                                                                                                                                                                                                                                                                                                            ;
    (l3-gamma3)*theta3rdot^2*cos(theta3r)+gamma4*theta4rdot^2*cos(theta4r)	                                                                                                                                                                                                                                                                                                                                                            ;
    -m1*g*gamma1*sin(theta1l)-m2*g*(l1)*sin(theta1l)-k2*(theta1l+phi2l-theta2l)-c2*(theta1ldot-theta2ldot)+k1*(theta+phi1l-theta1l)+c1*(thetadot-theta1ldot)	                                                                                                                                                                                                                                                                            ;
    -m2*g*gamma2*sin(theta2l)+k2*(theta1l+phi2l-theta2l)+c2*(theta1ldot-theta2ldot)	                                                                                                                                                                                                                                                                                                                                                        ;
    -m3*g*gamma3*sin(theta3l)-m4*g*(l3)*sin(theta3l)-k4*(theta3l+phi4l-theta4l)-c4*(theta3ldot-theta4ldot)+k3*(theta+phi3l-theta3l)+c3*(thetadot-theta3ldot)+fFl*(l3)*cos(theta3l)+fNl*(l3)*sin(theta3l)	                                                                                                                                                                                                                        ;
    -m4*g*gamma4*sin(theta4l)+k4*(theta3l+phi4l-theta4l)+c4*(theta3ldot-theta4ldot)+fFl*(l4)*cos(theta4l)+fNl*(l4)*sin(theta4l)	                                                                                                                                                                                                                                                                                                    ;
    gamma*thetadot^2*sin(theta)-gamma1*theta1ldot^2*sin(theta1l)	                                                                                                                                                                                                                                                                                                                                                                    ;
    -gamma*thetadot^2*cos(theta)+gamma1*theta1ldot^2*cos(theta1l)	                                                                                                                                                                                                                                                                                                                                                                    ;
    -(l1-gamma1)*theta1ldot^2*sin(theta1l)-gamma2*theta2ldot^2*sin(theta2l)	                                                                                                                                                                                                                                                                                                                                                            ;
    (l1-gamma1)*theta1ldot^2*cos(theta1l)+gamma2*theta2ldot^2*cos(theta2l)	                                                                                                                                                                                                                                                                                                                                                            ;
    -(l-gamma)*thetadot^2*sin(theta)-gamma3*theta3ldot^2*sin(theta3l)	                                                                                                                                                                                                                                                                                                                                                                ;
    (l-gamma)*thetadot^2*cos(theta)+gamma3*theta3ldot^2*cos(theta3l)	                                                                                                                                                                                                                                                                                                                                                                ;
    -(l3-gamma3)*theta3ldot^2*sin(theta3l)-gamma4*theta4ldot^2*sin(theta4l)	                                                                                                                                                                                                                                                                                                                                                            ;
    (l3-gamma3)*theta3ldot^2*cos(theta3l)+gamma4*theta4ldot^2*cos(theta4l)];

all_equations = M*state == C;

simplified_equations = eliminate(all_equations,[x1rdotdot y1rdotdot
    x2rdotdot y2rdotdot
    x3rdotdot y3rdotdot
    x4rdotdot y4rdotdot
    x1ldotdot y1ldotdot
    x2ldotdot y2ldotdot
    x3ldotdot y3ldotdot
    x4ldotdot y4ldotdot]);




