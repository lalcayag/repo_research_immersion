#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: konstantinos
this code produces the points of one beam that turns clockwise under the influence of a mean velocity
"""

import math 
import matplotlib.pyplot as plt

def beam(L_x,L_y,u_mean,beam_orig,angle_start,angle_stop,d_r,d_phi,time_step,r_max,r_min):
    x_range=[0,L_x]
    y_range=[0,L_y]
    rotations=0#rotations counter
    beam_points=[]#initialize the list that holds the point sets of all the beams
    for phi in range(angle_start,angle_stop,d_phi):
        rotations=rotations+1
        beam_orig[0]=beam_orig[0]+u_mean*time_step#translate the origin of the beam based on the velocity 
        phi=math.radians(phi)#convert angle phi to radians 
        point_x=[r_min*math.sin(phi)+beam_orig[0]]#calulate initial point
        point_y=[r_min*math.cos(phi)+beam_orig[1]]
        #while the next point is inside the domain continue 
        while (point_x[-1]>x_range[0] and point_x[-1]<x_range[1]) and (point_y[-1]>y_range[0] and point_y[-1]<y_range[1]):
            next_point_x=point_x[-1]+d_r*math.sin(phi)
            next_point_y=point_y[-1]+d_r*math.cos(phi)
            point_x.append(next_point_x)
            point_y.append(next_point_y)
        #remove last entity because it represents the last point wich is just ousite the domain     
        del point_x[-1]
        del point_y[-1]
        #create a list that keeps x and y cordinates of all the points of a certain beam
        point=[point_x,point_y]   
        #append this list to the general one
        beam_points.append(point)
    return  beam_points

#Define the case 
L_x,L_y=2000,1000
u_mean=35
beam_orig,angle_start,angle_stop=[0,0],20,88
d_r,d_phi=5,2
time_step=1#time that needs the scanner to rotate 
r_max,r_min=250,10


beam_points=beam(L_x,L_y,u_mean,beam_orig,angle_start,angle_stop,d_r,d_phi,time_step,r_max,r_min)

#plot a certain beam line
plt.plot(beam_points[-1][0][:], beam_points[-1][1][:], 'ro')#here you can change the first index to see different beam lines
plt.axis([0,L_x, 0, L_y])
plt.show()