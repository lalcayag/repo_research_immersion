# -*- coding: utf-8 -*-

# In[]
"""
@author: lalc
"""
import numpy as np
import scipy as sp
import math 
import matplotlib.pyplot as plt
import os
import tkinter as tkint
import tkinter.filedialog

from os import listdir
from os.path import isfile, join
#km:importing functions from folders
#answer la: yes""" 
import ppiscanprocess.windfieldrec as wr
import ppisynthetic.synthetic_wf_scan_noise_2 as sy
import ppiscanprocess.spectra_construction as sc

#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True

import pickle

import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

# In[]
class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

fmt = FormatScalarFormatter("%.2f")

def fm(x, pos=None):
    return r'${}$'.format('{:.2f}'.format(x).split('f')[0])
# In[Input files]

root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir')
root.destroy()

root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim Output dir')
root.destroy()


cwd = os.getcwd()
os.chdir(file_in_path)

onlyfiles = [f for f in listdir(file_in_path) if isfile(join(file_in_path, f))]

# In[]
##### Geometry definition (both synthetic and synthetic after reconst.) #####
# Grid points in Cartesian X-Y (2**n)
N_x = 2048
N_y = 2048

# Mean wind speed and Direction
#Dir = np.linspace(90,270,7)*np.pi/180
#Dir = [0*np.pi/180]
#km: a vector of 7 directions from 90 to 270 deg in rads
"""answer la: yes""" 

#u_mean = 15
 #km: the mean wind speed
"""answer la: yes, we can change this and make an array of wind speeds""" 

# Scan 0 geometry input
# rmin0,rmax0,nr0,phimin0,phimax0,np0,orig0 
"""km:definition of the scaner 0 minimum and maximum radial distance
minimum and maximum azimuth angle origin as an array x_0,y_0"""
"""answer la: yes""" 
""" la9: Now the azimuth angles and locations are transform to cartesian right away y-axis = N-S"""
rmin0,rmax0,nr0,phimin0,phimax0,np0,orig0 = 105,7000,198,90-256,90-344,45,np.array([0,6322832.3])
rp0 = (rmin0,rmax0,nr0,phimin0,phimax0,np0,orig0)
#km: tuple that contains the definition of the scanner0
"""answer la: yes""" 

# Scan 1 geometry input
# rmin1,rmax1,nr1,phimin1,phimax1,np1,orig1
rmin1,rmax1,nr1,phimin1,phimax1,np1,orig1 = 105,7000,198,90-284,90-196,45,np.array([0,6327082.4])
rp1 = (rmin1,rmax1,nr1,phimin1,phimax1,np1,orig1)

# Grids, polar and cartesian
d = orig1-orig0
#km: since they have the same y d[0] holds the distance of the scanners
"""answer la: yes""" 

# Polar grids for Scan 0 (local and translated)
r_0_g, phi_0_g, r_0_t, phi_0_t = sy.geom_polar_grid(rmin0,rmax0,nr0,phimin0,phimax0,np0,d)

# Polar grids for Scan 1 (local and translated)
r_1_g, phi_1_g, r_1_t, phi_1_t = sy.geom_polar_grid(rmin1,rmax1,nr1,phimin1,phimax1,np1,-d)

#plt.figure()
#plt.scatter((r_0_t*np.cos(phi_0_t)).flatten(),(r_0_t*np.sin(phi_0_t)).flatten())
#plt.scatter((r_1_t*np.cos(phi_1_t)).flatten(),(r_1_t*np.sin(phi_1_t)).flatten())

# In[]

#L_x, L_y, grid, x, y, tri, grid_new, d = sy.geom_syn_field(rp0, rp1, N_x, N_y) 
#
#print("L_x=",L_x,"L_y=",L_y)
##km4: Return the size of the in general rectangular (but now square) domain. A structured cartesian grid with N_x x N_y points. The coordinates of the grid points x y.
##km4: Another structured uniform grid for the same domain but with different spacing and the distance of the two scanners d. 
#"""answer la4: yes. grid_new is used as the recangular grid for wind field reconstruction from the values of V_LOS of each scan interpolated
#to this grid, if you see below (line 153), from grid_new phi_tri_1_s is calculated as de local (local to each Windscanner) azimuth angle used for reconstruction""" 
##km5:commented this line _,tri_i,_, _, _, _, _, _ = wr.grid_over2((r_1_g, np.pi-phi_1_g),(r_0_g, np.pi-phi_0_g),-d)
##km4: returns the trianguulation of the intersection set centers between the 2 scanners in cartesian coordinates
##km4: If I am not wrong, this procedure is also done in the sy.geom_syn_field function 
#"""answer la4: yes, indeed it is not used afterwards,  (you can erase this line I think)"""
##km6: grid is not used only grid_new is used in the rest of the code 
## Triangulation and weights for each scan
#dl = 75
#
## From Cartesian coord. to polar in global grid 
#r_tri_s = np.sqrt(grid_new[0]**2 + grid_new[1]**2)
#phi_tri_s = np.arctan2(grid_new[1],grid_new[0])
#r_tri_1_s, phi_tri_1_s = wr.translationpolargrid((r_tri_s, phi_tri_s),d/2)
#r_tri_0_s, phi_tri_0_s = wr.translationpolargrid((r_tri_s, phi_tri_s),-d/2)
#"""answer la4: So this wis just step to recover the original azimuth angle for each scan (local coordinates for each scan)
#              this time in the corresponding points of the reconstructed wind field in cartesian coordinates,
#              to be used in wind field reconstruction""" 
##km6: but again the results of this part (r_tri_0_s,r_tri_1_s etc) are never used in the code.
#"""answer la6: phi_tri_0_s and phi_tri_1_s is used for reconstruction""" 
#
## Mann-model parameters
#ae = np.array([0.025])#km5: create a variety of cases with a bunch of mann parameters. I will only run one case at least as a starting point. Do you recommend any parameters?
#L = np.array([50])
#G = np.array([2.0])
#seed =np.array([10])#km5: what does it represent?
#ae,L,G,seed = np.meshgrid(ae,L,G,-seed)
#sym = []
#no_sym = []
#geom_param0 = []

#############################################################
# In[]
""" This is the testing of the new functions I wrote and that
    you can use or compare to the ones you have writen"""
##############################################################
"""la9(?): Testing eom_syn_field2() and earlyweights2()"""

# A plotting function
def plot_mesh(x,y,x0,y0,title):
    import pylab as pl
    from matplotlib.collections import LineCollection

    pl.figure(figsize=(8, 8))
    
    hlines = np.column_stack(np.broadcast_arrays(x[0], y, x[-1], y))
    vlines = np.column_stack(np.broadcast_arrays(x, y[0], x, y[-1]))
    lines = np.concatenate([hlines, vlines]).reshape(-1, 2, 2)
    line_collection = LineCollection(lines, color="grey", linewidths=1,alpha=.3)
    ax = pl.gca()
    ax.set_aspect('equal')
    ax.use_sticky_edges = False
    ax.margins(0.07)
    ax.set_title(title)
    ax.add_collection(line_collection)
    plt.plot([x0[-1],x0[0]], [y0[0],y0[0]], color='grey')
    plt.plot([x0[0],x0[0]], [y0[0],y0[-1]], color='grey')
    plt.plot([x0[0],x0[-1]], [y0[-1],y0[-1]], color='grey')
    plt.plot([x0[-1],x0[-1]], [y0[-1],y0[0]], color='grey')
    return []

dir_mean = np.array([0, 90, 45, 270, 180])*np.pi/180
""" to cartesian"""
dir_mean = np.pi/2-dir_mean
dir_tit = wrap_angle((np.pi/2-dir_mean))*180/np.pi
u_mean = [0,15,100]
rot = 2 #2 degrees per second, rotational speed of the scan or scanning speed

L_x0,L_y0,grid0,x0,y0,tri0,_,_,x_0_rta0,y_0_rta0,x_1_rta0,y_1_rta0,center0 = sy.geom_syn_field2(rp0, rp1,
                                                                                        N_x, N_y, u_mean[0], rot, dir_mean[0],tri_ret = False)
L_x1,L_y1,grid1,x1,y1,tri1,_,_,x_0_rta1,y_0_rta1,x_1_rta1,y_1_rta1,center1 = sy.geom_syn_field2(rp0, rp1,
                                                                                        N_x, N_y, u_mean[1], rot, dir_mean[0],tri_ret = False)
L_x2,L_y2,grid2,x2,y2,tri2,_,_,x_0_rta2,y_0_rta2,x_1_rta2,y_1_rta2,center2 = sy.geom_syn_field2(rp0, rp1,
                                                                                        N_x, N_y, u_mean[1], rot, dir_mean[4],tri_ret = False)
L_x3,L_y3,grid3,x3,y3,tri3,_,_,x_0_rta3,y_0_rta3,x_1_rta3,y_1_rta3,center3 = sy.geom_syn_field2(rp0, rp1,
                                                                                        N_x, N_y, u_mean[1], rot, dir_mean[1], tri_ret = False)
L_x4,L_y4,grid4,x4,y4,tri4,_,_,x_0_rta4,y_0_rta4,x_1_rta4,y_1_rta4,center4 = sy.geom_syn_field2(rp0, rp1,
                                                                                        N_x, N_y, u_mean[1], rot, dir_mean[3], tri_ret = False)
L_x5,L_y5,grid5,x5,y5,tri5,_,_,x_0_rta5,y_0_rta5,x_1_rta5,y_1_rta5,center5 = sy.geom_syn_field2(rp0, rp1,
                                                                                        N_x, N_y, u_mean[1], rot, dir_mean[2], tri_ret = False)
L_x6,L_y6,grid6,x6,y6,tri6,_,_,x_0_rta6,y_0_rta6,x_1_rta6,y_1_rta6,center6 = sy.geom_syn_field2(rp0, rp1,
                                                                                        N_x, N_y, u_mean[2], rot, dir_mean[2], tri_ret = False)

plot_mesh(x0[::16],y0[::16],x0, y0, title = '$U _{mean}$ ='+'%.2f' %u_mean[0]+', Direction = '+'%.2f' %(dir_tit[0]))
plt.scatter(x_0_rta0.flatten(),y_0_rta0.flatten())
plt.scatter(x_1_rta0.flatten(),y_1_rta0.flatten())

plot_mesh(x1[::16],y1[::16],x1, y1, title = '$U _{mean}$ ='+'%.2f' %u_mean[1]+', Direction = '+'%.2f' %(dir_tit[0]))
plt.scatter(x_0_rta1.flatten(),y_0_rta1.flatten())
plt.scatter(x_1_rta1.flatten(),y_1_rta1.flatten())

plot_mesh(x2[::16],y2[::16],x2, y2, title = '$U _{mean}$ ='+'%.2f' %u_mean[1]+', Direction = '+'%.2f' %(dir_tit[4]))
plt.scatter(x_0_rta2.flatten(),y_0_rta2.flatten())
plt.scatter(x_1_rta2.flatten(),y_1_rta2.flatten())

plot_mesh(x3[::16],y3[::16],x3, y3, title = '$U _{mean}$ ='+'%.2f' %u_mean[1]+', Direction = '+'%.2f' %(dir_tit[1]))
plt.scatter(x_0_rta3.flatten(),y_0_rta3.flatten())
plt.scatter(x_1_rta3.flatten(),y_1_rta3.flatten())

plot_mesh(x4[::16],y4[::16],x4, y4, title = '$U _{mean}$ ='+'%.2f' %u_mean[1]+', Direction = '+'%.2f' %(dir_tit[3]))
plt.scatter(x_0_rta4.flatten(),y_0_rta4.flatten())
plt.scatter(x_1_rta4.flatten(),y_1_rta4.flatten())

plot_mesh(x5[::16],y5[::16],x5, y5, title = '$U _{mean}$ ='+'%.2f' %u_mean[1]+', Direction = '+'%.2f' %(dir_tit[2]))
plt.scatter(x_0_rta5.flatten(),y_0_rta5.flatten())
plt.scatter(x_1_rta5.flatten(),y_1_rta5.flatten())

plot_mesh(x6[::16],y6[::16],x6, y6, title = '$U _{mean}$ ='+'%.2f' %u_mean[2]+', Direction = '+'%.2f' %(dir_tit[2]))
plt.scatter(x_0_rta6.flatten(),y_0_rta6.flatten())
plt.scatter(x_1_rta6.flatten(),y_1_rta6.flatten())
######################################################################
# In[YOur results come from here!!!!!!!!!!!!!!!!!!!!!]
######################################################################
""" The above was just to try diffrent speeds and the efect on the sampling positions,
here you can find the real testing, woth the final reconstruction
""" 
i = 0
j = 2
tri_calc = False
# You can change this list
dir_mean = np.array([0, 90, 45, 270, 180])*np.pi/180
""" to cartesian"""
dir_mean = np.pi/2-dir_mean
dir_tit = (np.pi/2-dir_mean)*180/np.pi
u_mean = [0,20,100]
rot = 2 #2 degrees per second, rotational speed of the scan or scanning speed
L_x, L_y, grid, x, y, tri, grid_new, d, x_0_rta, y_0_rta, x_1_rta, y_1_rta, center = sy.geom_syn_field2(rp0, rp1,         
                                                                                                        N_x, N_y, u_mean[j], rot, dir_mean[i],tri_ret = tri_calc)

#plot_mesh(x[::16],y[::16],x, y, title = '$U _{mean}$ ='+'%.2f' %u_mean[j]+', Direction = '+'%.2f' %(dir_tit[i]))
#plt.scatter(x_0_rta.flatten(),y_0_rta.flatten())
#plt.scatter(x_1_rta.flatten(),y_1_rta.flatten())

# The dir where the simulations will be saved or read
root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir')
root.destroy()
################################################################################
# Wind Field generation, you can insert a loop
u, v = sy.wind_sim(.025, 300, 3.5, -1, 2048, 2048, L_x, L_y, file_in_path, pre = 'res_im')

u = np.reshape(u,grid[0].shape).T
v = np.reshape(v,grid[0].shape).T
################################################################################
#Or you can read it from a file
#u = np.reshape(np.fromfile(file_in_path+'/res_imu3003.50.025-1', dtype=np.float32),(N_x,N_y)).T
#v = np.reshape(np.fromfile(file_in_path+'/res_imv3003.50.025-1', dtype=np.float32),(N_x,N_y)).T

#Be careful with the mean wind speed you are using with no advection

U_in = u_mean[j]*np.ones(u.shape) + u
V_in = np.zeros(u.shape)+v

"""This field is just to try :)"""
#U_in = np.array([list(u_mean[1]*np.arange(0,u.shape[0])/u.shape[0]),]*u.shape[1]).T# + u
#V_in = np.zeros(u.shape)#v

vtx0, wts0, w0, c_ref0, s_ref0, shapes,uv0,r_refine0,phi_refine0 = sy.early_weights_pulsed2(r_0_g,phi_0_g, tri,
                                                          dl, dir_mean[i] , d/2,
                                                          center,rot*np.pi/180, u_mean[j], tri_calc = tri_calc)

vtx1, wts1, w1, c_ref1, s_ref1, shapes,uv1,r_refine1,phi_refine1 = sy.early_weights_pulsed2(r_1_g,phi_1_g, tri,
                                                          dl, dir_mean[i] , -d/2,
                                                          center,rot*np.pi/180, u_mean[j], tri_calc = tri_calc)

# Maybe you dn't want to plot this, it was just for cheking the overlapping are for different mean wind speeds
plot_mesh(x[::16],y[::16],x, y, title = '$U _{mean}$ ='+'%.2f' %u_mean[0]+', $Direction$ = '+'%.2f' %dir_tit[i])
plt.scatter(uv0[::32,0],uv0[::32,1],c = wrap_angle(np.arctan2(s_ref0.flatten()[::32],c_ref0.flatten()[::32]))*180/np.pi,cmap='jet')
plt.colorbar()
plot_mesh(x[::16],y[::16],x, y, title = '$U _{mean}$ ='+'%.2f' %u_mean[0]+', $Direction$ = '+'%.2f' %dir_tit[i])
plt.scatter(uv1[::32,0],uv1[::32,1],c = wrap_angle(np.arctan2(s_ref1.flatten()[::32],c_ref1.flatten()[::32]))*180/np.pi,cmap='jet')
plt.colorbar()
########################################
#This is just to verify the right sampling

#U0 = sp.interpolate.RectBivariateSpline(x, y, U_in.T)(uv0[:,0], uv0[:,1],grid=False)
#V0 = sp.interpolate.RectBivariateSpline(x, y, V_in.T)(uv0[:,0], uv0[:,1],grid=False)
#U0 = np.reshape(U0,c_ref0.shape)
#V0 = np.reshape(V0,c_ref0.shape)
#V_L0 = c_ref0*U0 + s_ref0*V0 
#
#n0 = shapes[2]
#m0 = shapes[3]
##Weight Normalization
#w0 = w0/np.reshape(np.repeat(np.sum(w0,axis=1),w0.shape[1]),w0.shape)
#VLw0 = np.zeros((V_L0.shape[0],int((V_L0.shape[1]-1)/(n0-1))))
#for i in range(V_L0.shape[0]):
#    VLw0[i,:] = np.dot(w0,np.where(np.isnan(V_L0.T[:,i]),0,V_L0.T[:,i]))  
#w_p0 = np.ones(VLw0.shape)/(m0-1) 
#VLw00 = (VLw0[:-1,:]*w_p0[:-1,:]) 
#
#VLw000 = np.nansum(VLw00.reshape(-1,(m0-1),VLw00.shape[-1]),axis=1)
#
#U1 = sp.interpolate.RectBivariateSpline(x, y, U_in)(uv1[:,0], uv1[:,1],grid=False)
#V1 = sp.interpolate.RectBivariateSpline(x, y, V_in)(uv1[:,0], uv1[:,1],grid=False)
#U1 = np.reshape(U1,c_ref1.shape)
#V1 = np.reshape(V1,c_ref1.shape)
#V_L1 = c_ref1*U1 + s_ref1*V1 
#plot_mesh(x[::16],y[::16],x, y, title = '$U_{mean}$ ='+'%.2f' %u_mean[0]+', $Direction$ = '+'%.2f' %dir_tit[i])
#plt.scatter(uv0[::32,0],uv0[::32,1],c = V_L0.flatten()[::32], cmap='jet')
#plt.colorbar()
#plot_mesh(x[::16],y[::16],x, y, title = '$U_{mean}$ ='+'%.2f' %u_mean[0]+', $Direction$ = '+'%.2f' %dir_tit[i])
#plt.scatter(uv1[::32,0],uv1[::32,1],c = V_L1.flatten()[::32], cmap='jet')
#plt.colorbar()
#########################################
##Numerical lidar sampling

if tri_calc:
    vlos0 = sy.num_pulsed_lidar(U_in,V_in,vtx0,wts0,w0,c_ref0, s_ref0, shapes)
    vlos1 = sy.num_pulsed_lidar(U_in,V_in,vtx1,wts1,w1,c_ref1, s_ref1, shapes)
else:
    vlos0 = sy.num_pulsed_lidar2(U_in,V_in,x,y,uv0,w0,c_ref0, s_ref0, shapes)
    vlos1 = sy.num_pulsed_lidar2(U_in,V_in,x,y,uv1,w1,c_ref1, s_ref1, shapes) 

#Interpolation to cartesian grid
vlos0_int_sq = sp.interpolate.griddata(np.c_[(r_0_t*np.cos(phi_0_t)).flatten(),
                                             (r_0_t*np.sin(phi_0_t)).flatten()],
                                             vlos0.flatten(), (grid_new[0].flatten(),
                                             grid_new[1].flatten()), method='cubic')
vlos1_int_sq = sp.interpolate.griddata(np.c_[(r_1_t*np.cos(phi_1_t)).flatten(),
                                             (r_1_t*np.sin(phi_1_t)).flatten()],
                                             vlos1.flatten(), (grid_new[0].flatten(),
                                             grid_new[1].flatten()), method='cubic')

vlos0_int_sq = np.reshape(vlos0_int_sq,grid_new[0].shape)
vlos1_int_sq = np.reshape(vlos1_int_sq,grid_new[0].shape)

r_tri_s = np.sqrt(grid_new[0]**2 + grid_new[1]**2)
phi_tri_s = np.arctan2(grid_new[1],grid_new[0])
_, phi_tri_1_s = wr.translationpolargrid((r_tri_s, phi_tri_s),d/2)
_, phi_tri_0_s = wr.translationpolargrid((r_tri_s, phi_tri_s),-d/2)

phi0sq = phi_tri_0_s
phi0sq[np.isnan(vlos0_int_sq)] = np.nan

phi1sq = phi_tri_1_s
phi1sq[np.isnan(vlos1_int_sq)] = np.nan

Ur,Vr = sy.dir_rec_rapid(vlos0_int_sq.flatten(),vlos1_int_sq.flatten(),
                         wrap_angle(phi0sq).flatten(),wrap_angle(phi1sq).flatten(),
                         grid_new[0].shape)

# If you want to plot to check, is up to you
plt.figure()
plt.contourf(grid_new[0], grid_new[1], Ur, cmap='jet')
plt.colorbar()
plt.figure()
plt.contourf(grid_new[0], grid_new[1], Vr, cmap='jet')
plt.colorbar()

#########################
# If you want to check the autocorrelatiobn of the original wind field
#########################
U_o = np.reshape(sp.interpolate.RectBivariateSpline(x, y, U_in.T)(grid_new[0].flatten(),
                                         grid_new[1].flatten(),grid=False),grid_new[0].shape)
V_o = np.reshape(sp.interpolate.RectBivariateSpline(x, y, V_in.T)(grid_new[0].flatten(),
                                         grid_new[1].flatten(),grid=False),grid_new[0].shape)
#########################
#########################
#
#overlap = np.isnan(vlos0_int_sq) | np.isnan(vlos1_int_sq) 
#
#U_o[overlap] = np.nan
#V_o[overlap] = np.nan

# Some plots that could be useful
#plt.figure()
#plt.contourf(r_0_t*np.cos(phi_0_t), r_0_t*np.sin(phi_0_t), vlos0,30, cmap='jet')
#plt.colorbar()
#plt.figure()
#plt.contourf(r_1_t*np.cos(phi_1_t), r_1_t*np.sin(phi_1_t), vlos1, 30, cmap='jet')
#plt.colorbar()
#
#plt.figure()
#plt.contourf(grid_new[0], grid_new[1], vlos0_int_sq, cmap='jet')
#plt.contourf(grid_new[0], grid_new[1], vlos1_int_sq, cmap='jet')
#plt.colorbar()
#
#plt.figure()
#plt.contourf(grid_new[0], grid_new[1], wrap_angle(phi0sq)*180/np.pi, 20, cmap='jet')
#plt.colorbar()
#plt.contourf(grid_new[0], grid_new[1], wrap_angle(phi1sq)*180/np.pi, 20, cmap='rainbow')
#plt.colorbar()

#plt.figure()
#plt.contourf(grid_new[0], grid_new[1], U_o, cmap='jet')
#plt.colorbar()
#plt.figure()
#plt.contourf(grid_new[0], grid_new[1], V_o, cmap='jet')
#plt.colorbar()


#########################
# Autocorreltion and length scales
U_mean = np.nanmean(Ur.flatten())
V_mean = np.nanmean(Vr.flatten())
gamma = np.arctan2(V_mean,U_mean)
tau,eta,r_u,r_v,r_uv,_,_,_,_ = sc.spatial_autocorr_sq(grid_new,Ur,Vr, 
                            transform = False, transform_r = True,gamma=gamma,e_lim=.1,refine=32)
Lu = np.array(sc.integral_lenght_scale(r_u,tau,eta))
Lv = np.array(sc.integral_lenght_scale(r_v,tau,eta))
#########################

#####################################
""" end of testing (not testing anymore though)"""
#####################################




# In[]

for dir_mean in Dir:#km5: for each direction. Do you generate different realizations by rotatiing the scanners ?  
  
    vtx0, wts0, w0, c_ref0, s_ref0, shapes,uv0_t,vu_beam_0,uv0 = sy.early_weights_pulsed(r_0_g,np.pi-phi_0_g, dl, dir_mean , tri, -d/2, y[0]/2,orig0,0,L_x,L_y)#km5: pass the local polar coordinates of the scanner0 
    vtx1, wts1, w1, c_ref1, s_ref1, shapes,uv1_t,vu_beam_1,uv1 = sy.early_weights_pulsed(r_1_g,np.pi-phi_1_g, dl, dir_mean , tri, d/2, y[0]/2,orig1,1,L_x,L_y)
    #store data
    geom_param0.append((vtx0, wts0, w0, c_ref0, s_ref0, shapes))
    geom_param0.append((vtx1, wts1, w1, c_ref1, s_ref1, shapes))
    
    Urec = []
    Vrec = []
    print(dir_mean*180/np.pi,u_mean)
    for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
            
        if (L_i == 62.5): 
            u_file_name = 'simu'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            v_file_name = 'simv'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)

        else:
            u_file_name = 'simu'+str(int(L[0][0][0]))+str(float(G[0][0][0]))+str(float(ae[0][0][0]))+str(int(seed[0][0][0]))
            v_file_name = 'simv'+str(int(L[0][0][0]))+str(float(G[0][0][0]))+str(float(ae[0][0][0]))+str(int(seed[0][0][0]))
            
            
        if u_file_name in onlyfiles:
            print('yes')
            sym.append([ae_i,L_i,G_i,seed_i])
            u = np.reshape(np.fromfile(u_file_name, dtype=np.float32),(N_x,N_y)).T#km5:reading the fluctuations of velocity component x
            v = np.reshape(np.fromfile(v_file_name, dtype=np.float32),(N_x,N_y)).T#km5:reading the fluctuations of velocity component y
            U_in = u_mean + u#km5:adding the mean velocity component x to the fluctuations
            V_in = 0 + v
            #Numerical lidar sampling
            vlos0 = sy.num_pulsed_lidar(U_in,V_in,vtx0,wts0,w0,c_ref0, s_ref0, shapes)
            #km6: takes as an input the vertices and the weights and returns the vlos in polar coordinates 
            """answer la6: yes""" 
            vlos1 = sy.num_pulsed_lidar(U_in,V_in,vtx1,wts1,w1,c_ref1, s_ref1, shapes)
       
            #Interpolation to cartesian grid
            vlos1_int_sq = sp.interpolate.griddata(np.c_[(r_1_t*np.cos(phi_1_t)).flatten(),(r_1_t*np.sin(phi_1_t)).flatten()],vlos1.flatten(), (grid_new[0].flatten(), grid_new[1].flatten()), method='cubic')
            vlos0_int_sq = sp.interpolate.griddata(np.c_[(r_0_t*np.cos(phi_0_t)).flatten(),(r_0_t*np.sin(phi_0_t)).flatten()],vlos0.flatten(), (grid_new[0].flatten(), grid_new[1].flatten()), method='cubic')
            
            vlos1_int_sq = np.reshape(vlos1_int_sq,grid_new[0].shape)
            #km6  reshapes the vector matrix to square (if the domain changes should we change it or not ??)
            """answer la6: No, it is reshaped because comes as a 1d array"""
            vlos0_int_sq = np.reshape(vlos0_int_sq,grid_new[0].shape)
            
            #Wind field reconstruction (overlaping are of the two scans)
            U,V = sy.dir_rec_rapid(vlos1_int_sq.flatten(),vlos0_int_sq.flatten(), phi_tri_1_s.flatten(),phi_tri_0_s.flatten(),grid_new[0].shape)
            #km6: why u has small values while v has values close to umean=15?
            """answer la6: If the direction of the wind is in the axis west-east, then goes along the y local coordinate, or V"""
                
            #Storing
            
            vlos0_file_name = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            vlos1_file_name = 'vlos1'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            U_file_name = 'U'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            V_file_name = 'V'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            (vlos0.flatten()).astype(np.float32).tofile(vlos0_file_name)
            (vlos1.flatten()).astype(np.float32).tofile(vlos1_file_name)
            (U.flatten()).astype(np.float32).tofile(U_file_name)
            (V.flatten()).astype(np.float32).tofile(V_file_name)
        else:
            print('no')
            no_sym.append([ae_i,L_i,G_i,seed_i])

odd = [1,3,5,7,9,11,13]
even = [0,2,4,6,8,10,12]
#
#for i,j,direction in zip(odd,even,Dir):
#    with open('geom_param0'+str(int(direction*180/np.pi))+'.pkl', 'wb') as geom:
#     pickle.dump(geom_param0[j],geom)
#     
#    with open('geom_param1'+str(int(direction*180/np.pi))+'.pkl', 'wb') as geom:
#     pickle.dump(geom_param0[i],geom) 
    
#with open('sim.pkl', 'wb') as sim:
 #    pickle.dump((no_sym,sym),sim)

# In[Auto and cross-correlation]
####################################################################################################################################
## Comment for Konstantinos: Autocorrelation, very slow, but works with any geometry
####################################################################################################################################    
"""answer la7: try this first""" 
onlyfiles = [f for f in listdir(file_in_path) if isfile(join(file_in_path, f))]  
x = grid_new[0][0,:]
y = grid_new[1][:,0]  
dx = np.diff(x)[0] 
dy = np.diff(y)[0] 
count=0
symr = []
ae = np.array([0.025])#km5: create a variety of cases with a bunch of mann parameters. I will only run one case at leadt as a starting point. Do you recommend any parameters?
L = np.array([50])
G = np.array([2.0])
seed =np.array([10])#km5: what does it represent?
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)
length_scales=[]
for dir_mean in Dir:
    trical = True
    valid_out = True
    print(dir_mean*180/np.pi,u_mean)
    for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):                       
        U_file_name = 'U'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        V_file_name = 'V'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        r_uv_name = 'r_uv'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)  
        if (U_file_name in onlyfiles):# & (~(r_uv_name in onlyfiles)):
            symr.append([dir_mean*180/np.pi,ae_i,L_i,G_i,seed_i])
            U = np.reshape(np.fromfile(U_file_name, dtype=np.float32),grid_new[0].shape)
            V = np.reshape(np.fromfile(V_file_name, dtype=np.float32),grid_new[0].shape)
            U_mean = np.nanmean(U.flatten())
            V_mean = np.nanmean(V.flatten())
            gamma = np.arctan2(V_mean,U_mean)
            tau,eta,r_u,r_v,r_uv,valid,indicator,e,egrad = sc.spatial_autocorr_sq(grid_new,U,V, transform = False, transform_r = True,gamma=gamma,e_lim=.08,refine=32)         
            tau_name = 'tau'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            eta_name = 'eta'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            r_u_name = 'r_u'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            r_v_name = 'r_v'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            r_uv_name = 'r_uv'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            
            (tau.flatten()).astype(np.float32).tofile(tau_name)
            (eta.flatten()).astype(np.float32).tofile(eta_name)
            (r_u.flatten()).astype(np.float32).tofile(r_u_name)
            (r_v.flatten()).astype(np.float32).tofile(r_v_name)
            (r_uv.flatten()).astype(np.float32).tofile(r_uv_name)          
            print(symr[count])
            count+=1
            
            
with open('simr.pkl', 'wb') as sim:
     pickle.dump(symr,sim)
# In[Spectra from autocorrelation and fft]
# interpolation two binary grid
####################################################################################################################################
## Comment for Konstantinos: Spectra from autocorrelation, if you have it
####################################################################################################################################    
"""answer la6: and this""" 
x_max = np.max(np.r_[(r_0_t*np.cos(phi_0_t)).flatten(),(r_1_t*np.cos(phi_1_t)).flatten()])
x_min = np.min(np.r_[(r_0_t*np.cos(phi_0_t)).flatten(),(r_1_t*np.cos(phi_1_t)).flatten()])

y_max = np.max(np.r_[(r_0_t*np.sin(phi_0_t)).flatten(),(r_1_t*np.sin(phi_1_t)).flatten()])
y_min = np.min(np.r_[(r_0_t*np.sin(phi_0_t)).flatten(),(r_1_t*np.sin(phi_1_t)).flatten()])

x_o = np.linspace(x_min,x_max,N_x)
y_o = np.linspace(y_min,y_max,N_y)   

n_tau, m_tau = 512,512
       
for dir_mean in Dir:
    trical = True
    print(dir_mean*180/np.pi,u_mean)
    for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):                       
        tau_name = 'tau'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        eta_name = 'eta'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        r_u_name = 'r_u'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        r_v_name = 'r_v'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        r_uv_name = 'r_uv'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        U_file_name = 'U'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        V_file_name = 'V'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i) 
        if ((U_file_name in onlyfiles) & ((r_uv_name in onlyfiles))):
            print([int(dir_mean*180/np.pi),ae_i,L_i,G_i,seed_i])
            U = np.reshape(np.fromfile(U_file_name, dtype=np.float32),grid_new[0].shape)
            V = np.reshape(np.fromfile(V_file_name, dtype=np.float32),grid_new[0].shape)
            tau = np.fromfile(tau_name, dtype=np.float32)
            eta = np.fromfile(eta_name, dtype=np.float32)
            r_u = np.fromfile(r_u_name, dtype=np.float32)
            r_v = np.fromfile(r_v_name, dtype=np.float32)
            r_uv = np.fromfile(r_uv_name, dtype=np.float32)                     
            tau_int = np.linspace(np.min(tau[tau>0]),np.max(tau[tau>0]),256)
            tau_int = np.r_[-np.flip(tau_int),0,tau_int]
            eta_int = np.linspace(np.min(eta[eta>0]),np.max(eta[eta>0]),256)
            eta_int = np.r_[-np.flip(eta_int),0,eta_int]
            tau_int, eta_int = np.meshgrid(tau_int,eta_int)                     
            _,_,ru_i = sc.autocorr_interp_sq(r_u, eta, tau, tau_lin = tau_int, eta_lin = eta_int)
            _,_,rv_i = sc.autocorr_interp_sq(r_v, eta, tau, tau_lin = tau_int, eta_lin = eta_int)
            _,_,ruv_i = sc.autocorr_interp_sq(r_uv, eta, tau, tau_lin = tau_int, eta_lin = eta_int)          
            ru_i[np.isnan(ru_i)]=0
            rv_i[np.isnan(rv_i)]=0
            ruv_i[np.isnan(ruv_i)]=0   
            ru_i[tau_int<0]=np.flip(ru_i[tau_int>0])
            rv_i[tau_int<0]=np.flip(rv_i[tau_int>0])           
            ru_i[eta_int<0]=np.flip(ru_i[eta_int>0])
            rv_i[eta_int<0]=np.flip(rv_i[eta_int>0])
            ku_r,kv_r,Su_r,Sv_r,Suv_r = sc.spectra_fft((tau_int,eta_int),ru_i,rv_i,ruv_i,K=0)                           
            ku_r_name = 'ku_r_name'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            kv_r_name = 'kv_r_name'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            Su_r_name = 'Su_r_name'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            Sv_r_name = 'Sv_r_name'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            Suv_r_name = 'Suv_r_name'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            print('savingSur')
            (ku_r.flatten()).astype(np.float32).tofile(ku_r_name)
            (kv_r.flatten()).astype(np.float32).tofile(kv_r_name)
            (np.real(Su_r).flatten()).astype(np.float32).tofile(Su_r_name)
            (np.real(Sv_r).flatten()).astype(np.float32).tofile(Sv_r_name)
            (np.real(Suv_r).flatten()).astype(np.float32).tofile(Suv_r_name)
            (np.imag(Suv_r).flatten()).astype(np.float32).tofile(Suv_r_name+'imag')            

             
# In[Transfer function]
####################################################################################################################################
## Comment for Konstantinos: Here the Filter (or transfer function in the frequency space) is estimated in one dimension
####################################################################################################################################    
#ae = np.array([0.025])#km5: create a variety of cases with a bunch of mann parameters. I will only run one case at leadt as a starting point. Do you recommend any parameters?
#L = np.array([50])
#G = np.array([2.0])
#seed =np.array([10])#km5: what does it represent?
#ae,L,G,seed = np.meshgrid(ae,L,G,-seed)

root = tkint.Tk()
file_in_path_r = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir')
root.destroy()

onlyfiles_r = [f for f in listdir(file_in_path_r) if isfile(join(file_in_path_r, f))] 

k_H = []
H = []
sym = []
for dir_mean in Dir:
    for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):            
        if (L_i == 62.5): 
            u_file_name = 'simu'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            v_file_name = 'simv'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        else:
            u_file_name = 'simu'+str(int(L_i))+str(G_i)+str(ae_i)+str(seed_i)
            v_file_name = 'simv'+str(int(L_i))+str(G_i)+str(ae_i)+str(seed_i)
        ku_r_name = 'ku_r_name'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        kv_r_name = 'kv_r_name'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        Su_r_name = 'Su_r_name'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        Sv_r_name = 'Sv_r_name'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        Suv_r_name = 'Suv_r_name'+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        #km5: U_file_name = 'U'+str(15)+str(int(Dir[i]*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        #km5: V_file_name = 'V'+str(15)+str(int(Dir[i]*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        U_file_name = 'U'+str(15)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        V_file_name = 'V'+str(15)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        
        if Suv_r_name in onlyfiles_r:        
            k_u_r = np.fromfile(join(file_in_path_r,ku_r_name), dtype=np.float32)
            k_v_r = np.fromfile(join(file_in_path_r,kv_r_name), dtype=np.float32)
            S_u_r = np.fromfile(join(file_in_path_r,Su_r_name), dtype=np.float32)
            S_v_r = np.fromfile(join(file_in_path_r,Sv_r_name), dtype=np.float32)
            S_uv_r= np.fromfile(join(file_in_path_r,Suv_r_name), dtype=np.float32)
            kur,kvr = np.meshgrid(k_u_r,k_v_r)
            S_u_r = np.reshape(S_u_r,kur.shape)
            S_v_r = np.reshape(S_v_r,kur.shape)
            S_uv_r = np.reshape(S_uv_r,kur.shape)          
            u = np.reshape(np.fromfile(join(file_in_path_r,u_file_name), dtype=np.float32),(N_x,N_y)).T
            v = np.reshape(np.fromfile(join(file_in_path_r,v_file_name), dtype=np.float32),(N_x,N_y)).T              
            #km5: k_u_o,k_v_o,S_u_o,S_v_o,S_uv_o = sc.spatial_spec_sq(x0,y0,np.flipud(np.reshape(u,(N_x,N_y)).T),np.flipud(np.reshape(v,(N_x,N_y)).T),transform = False, ring=False)
            k_u_o,k_v_o,S_u_o,S_v_o,S_uv_o = sc.spatial_spec_sq(x_o,y_o,np.flipud(np.reshape(u,(N_x,N_y)).T),np.flipud(np.reshape(v,(N_x,N_y)).T),transform = False, ring=False)
            Suo_ave=sc.spectra_average(S_u_o,(k_u_o, k_v_o),bins=20).S
            Svo_ave=sc.spectra_average(S_v_o,(k_u_o, k_v_o),bins=20).S
            Sur_ave=sc.spectra_average(S_u_r,(k_u_r, k_v_r),bins=20).S
            Svr_ave=sc.spectra_average(S_v_r,(k_u_r, k_v_r),bins=20).S            
            Su_o1D_ave = sp.integrate.simps(.5*(Suo_ave+Svo_ave),k_v_o,axis=0)
            Su_r1D_ave = sp.integrate.simps(.5*(Sur_ave+Svr_ave),k_v_r,axis=0)
            Su_o1D_ave_it = np.exp(sp.interpolate.interp1d(np.log(k_u_o[k_u_o>0]),
                            np.log(Su_o1D_ave[k_u_o>0]))(np.log(k_u_r[k_u_r>np.min(k_u_o[k_u_o>0])])))            
            k_H.append(k_u_r[k_u_r>np.min(k_u_o[k_u_o>0])])
            H.append(Su_r1D_ave[k_u_r>np.min(k_u_o[k_u_o>0])]/Su_o1D_ave_it)
            sym.append([dir_mean*180/np.pi,ae_i,L_i,G_i,seed_i])
            print(([dir_mean*180/np.pi,ae_i,L_i,G_i,seed_i]))
            
#with open('H.pkl', 'wb') as V_t:
#     pickle.dump((k_H,H,sym),V_t)

with open('H.pkl', 'rb') as V_t:
     k_H,H,sym = pickle.load(V_t)            

# In[]
####################################################################################################################################
## Comment for Konstantinos: This is the spectra form the Fourier transform applied to the velcity field.
####################################################################################################################################    

k_H_s = []
H_s = []
sym_s = []
for dir_mean in Dir:
    trical = True
    for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):            
        if (L_i == 62.5): 
            u_file_name = 'simu'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            v_file_name = 'simv'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        else:
            u_file_name = 'simu'+str(int(L_i))+str(G_i)+str(ae_i)+str(seed_i)
            v_file_name = 'simv'+str(int(L_i))+str(G_i)+str(ae_i)+str(seed_i)
        U_file_name = 'U'+str(15)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        V_file_name = 'V'+str(15)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        if U_file_name in onlyfiles_r:        
            U = np.reshape(np.fromfile(U_file_name, dtype=np.float32),grid_new[0].shape)
            V = np.reshape(np.fromfile(V_file_name, dtype=np.float32),grid_new[0].shape)
            if trical:
                _, _, mask, mask_int, tri_del = sc.field_rot(grid_new[0][0,:], grid_new[1][:,0], U, V, gamma=None, tri_calc = True)
                trical = False
            
            k_u_s,k_v_s,S_u_s,S_v_s,S_uv_s = sc.spatial_spec_sq(grid_new[0][0,:],grid_new[1][:,0],U,V,tri_del = tri_del, mask_int = mask_int, tri_calc = False, transform = True)

            u = np.reshape(np.fromfile(join(file_in_path_r,u_file_name), dtype=np.float32),(N_x,N_y)).T
            v = np.reshape(np.fromfile(join(file_in_path_r,v_file_name), dtype=np.float32),(N_x,N_y)).T              

            #km5: k_u_o,k_v_o,S_u_o,S_v_o,S_uv_o = sc.spatial_spec_sq(x0,y0,np.flipud(np.reshape(u,(N_x,N_y)).T),np.flipud(np.reshape(v,(N_x,N_y)).T),transform = False, ring=False)
            k_u_o,k_v_o,S_u_o,S_v_o,S_uv_o = sc.spatial_spec_sq(x_o,y_o,np.flipud(np.reshape(u,(N_x,N_y)).T),np.flipud(np.reshape(v,(N_x,N_y)).T),transform = False, ring=False)
 
"""answer la6: There are some things I havent modified here since the last time I used it, but check the main function,
 that can be messy and compare this results to the autocorrelation ones"""            
#            Suo_ave=sc.spectra_average(S_u_o,(k_u_o, k_v_o),bins=20).S
#            Svo_ave=sc.spectra_average(S_v_o,(k_u_o, k_v_o),bins=20).S
#            Sus_ave=sc.spectra_average(S_u_s,(k_u_s, k_v_s),bins=20).S
#            Svs_ave=sc.spectra_average(S_v_s,(k_u_s, k_v_s),bins=20).S     
"""answer la6: The average that you see in the spectra here is to calculate the spectra of horizontal fluctuations S_h,
were u and v are merge in one number. This assumes that the fluctuations are axisymmetric or isotropic, wghich is not the case ussualy, it is just a model
from Peltier 1996"""               
            Su_o1D_ave = sp.integrate.simps(.5*(Suo_ave+Svo_ave),k_v_o,axis=0)
            Su_s1D_ave = sp.integrate.simps(.5*(Sus_ave+Svs_ave),k_v_s,axis=0)
            
            Su_o1D_ave_it = np.exp(sp.interpolate.interp1d(np.log(k_u_o[k_u_o>0]),
                            np.log(Su_o1D_ave[k_u_o>0]))(np.log(k_u_s[k_u_s>np.min(k_u_o[k_u_o>0])])))            
            k_H_s.append(k_u_s[k_u_s>np.min(k_u_o[k_u_o>0])])
            H_s.append(Su_s1D_ave[k_u_s>np.min(k_u_o[k_u_o>0])]/Su_o1D_ave_it)
            sym_s.append([dir_mean*180/np.pi,ae_i,L_i,G_i,seed_i])
            print(([dir_mean*180/np.pi,ae_i,L_i,G_i,seed_i]))     

with open('H_s.pkl', 'rb') as V_t:
     k_H_s,H_s,sym_s = pickle.load(V_t)      
# In[]  
####################################################################################################################################
## Comment for Konstantinos: This is just a bunch of figures, not necessary you use them in your project
####################################################################################################################################    

lengths = [len(hi) for hi in k_H]

colors = ['b','r','g','k'] # directions
mark = ['o','s','^','v'] # quadrant

ind0 = (np.array(sym)[:,3] >= 2 ) & (np.array(sym)[:,2] >= 500 )
ind1 = (np.array(sym)[:,3] < 2 ) & (np.array(sym)[:,2] >= 500 )
ind2 = (np.array(sym)[:,3] >= 2 ) & (np.array(sym)[:,2] < 500 )
ind3 = (np.array(sym)[:,3] < 2 ) & (np.array(sym)[:,2] < 500 )

plt.figure()
for i, dir_mean in enumerate(Dir[[0,3]]):
    ind_dir = (np.array(sym)[:,0] == dir_mean*180/np.pi)
    ind_0 = ind_dir & ind0
    k_plot = np.mean(np.array(k_H)[ind_0,:],axis=0)
    S_plot = np.mean(np.array(H)[ind_0,:],axis=0)
    ind_plot = k_plot<5*10**-2
    plt.plot(k_plot[ind_plot],S_plot[ind_plot], c = colors[i], marker=mark[0], 
             label = str(int(dir_mean*180/np.pi))+' degrees' + 'L  $>=$ 500 and G $>=$ 2' )
    ind_1 = ind_dir & ind1
    k_plot = np.mean(np.array(k_H)[ind_1,:],axis=0)
    S_plot = np.mean(np.array(H)[ind_1,:],axis=0)
    ind_plot = k_plot<5*10**-2
    plt.plot(k_plot[ind_plot],S_plot[ind_plot], c = colors[i], marker=mark[1], 
             label = str(int(dir_mean*180/np.pi))+' degrees' + 'L $>=$ 500 and G $<$ 2' )
    ind_2 = ind_dir & ind2
    k_plot = np.mean(np.array(k_H)[ind_2,:],axis=0)
    S_plot = np.mean(np.array(H)[ind_2,:],axis=0)
    ind_plot = k_plot<5*10**-2
    plt.plot(k_plot[ind_plot],S_plot[ind_plot], c = colors[i], marker=mark[2], 
             label = str(int(dir_mean*180/np.pi))+' degrees' + 'L $<$ 500 and G $>=$ 2' )
    ind_3 = ind_dir & ind3
    k_plot = np.mean(np.array(k_H)[ind_3,:],axis=0)
    S_plot = np.mean(np.array(H)[ind_3,:],axis=0)
    ind_plot = k_plot<5*10**-2
    plt.plot(k_plot[ind_plot],S_plot[ind_plot], c = colors[i], marker=mark[3], 
             label = str(int(dir_mean*180/np.pi))+' degrees' + 'L $<$ 500 and G $<$ 2' )
    
    
#    plt.plot(np.mean(np.array(k_H)[ind,:],axis=0),np.mean(np.array(H)[ind,:],axis=0)+np.std(np.array(H)[ind,:],axis=0),'--')
#    plt.plot(np.mean(np.array(k_H)[ind,:],axis=0),np.mean(np.array(H)[ind,:],axis=0)-np.std(np.array(H)[ind,:],axis=0),'--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k1')
    plt.ylabel('H')
plt.legend()


# In[]     
lengths = [len(hi) for hi in k_H]

colors = ['b','r','g','k'] # directions
mark = ['o','s','^','v'] # quadrant

ind0 = (np.array(sym)[:,3] >= 2 ) & (np.array(sym)[:,2] >= 500 )
ind1 = (np.array(sym)[:,3] < 2 ) & (np.array(sym)[:,2] >= 500 )
ind2 = (np.array(sym)[:,3] >= 2 ) & (np.array(sym)[:,2] < 500 )
ind3 = (np.array(sym)[:,3] < 2 ) & (np.array(sym)[:,2] < 500 )

Su_o1D_ave_it = np.exp(sp.interpolate.interp1d(np.log(k_u_o[k_u_o>0]),
            np.log(Su_o1D_ave[k_u_o>0]))(np.log(k_u_s[k_u_s>np.min(k_u_o[k_u_o>0])])))  

plt.figure()
H_s_int = []
for i in range(len(H_s)):
    H_s_int.append(np.exp(sp.interpolate.interp1d(np.log(k_H_s[i]),
            np.log(H_s[i]))(np.log(k_H[i]))))  
    
# In[]    
    
colors = ['b','r','g','k'] # directions
mark = ['o','s','^','v'] # quadrant

ind0 = (np.array(sym)[:,3] >= 2 ) & (np.array(sym)[:,2] >= 500 )
ind1 = (np.array(sym)[:,3] < 2 ) & (np.array(sym)[:,2] >= 500 )
ind2 = (np.array(sym)[:,3] >= 2 ) & (np.array(sym)[:,2] < 500 )
ind3 = (np.array(sym)[:,3] < 2 ) & (np.array(sym)[:,2] < 500 )

plt.figure()
for i, dir_mean in enumerate(Dir[:4]):
    ind_dir = (np.array(sym)[:,0] == dir_mean*180/np.pi)
    ind_0 = ind_dir# & ind0
    k_plot = np.mean(np.array(k_H)[ind_0,:],axis=0)
    S_plot = np.mean(np.array(H_s_int)[ind_0,:],axis=0)
    
    S_plot_var = np.std(np.array(H_s_int)[ind_0,:],axis=0)
    
    ind_plot = k_plot<5*10**-2
    plt.plot(k_plot[ind_plot],S_plot[ind_plot], c = colors[i], #marker=mark[0], 
             label = str(int(dir_mean*180/np.pi))+' degrees' + 'L  $>=$ 500 and G $>=$ 2' )
    
    plt.plot(k_plot[ind_plot],S_plot[ind_plot]+S_plot_var[ind_plot],'--', c = colors[i], #marker=mark[0], 
             label = str(int(dir_mean*180/np.pi))+' degrees' + 'L  $>=$ 500 and G $>=$ 2' )
    
    plt.plot(k_plot[ind_plot],S_plot[ind_plot]-S_plot_var[ind_plot],'--', c = colors[i], #marker=mark[0], 
             label = str(int(dir_mean*180/np.pi))+' degrees' + 'L  $>=$ 500 and G $>=$ 2' )
    ind_1 = ind_dir & ind1
    k_plot = np.mean(np.array(k_H)[ind_1,:],axis=0)
    S_plot = np.mean(np.array(H_s_int)[ind_1,:],axis=0)
    ind_plot = k_plot<5*10**-2
    plt.plot(k_plot[ind_plot],S_plot[ind_plot], c = colors[i], marker=mark[1], 
             label = str(int(dir_mean*180/np.pi))+' degrees' + 'L $>=$ 500 and G $<$ 2' )
    ind_2 = ind_dir & ind2
    k_plot = np.mean(np.array(k_H)[ind_2,:],axis=0)
    S_plot = np.mean(np.array(H_s_int)[ind_2,:],axis=0)
    ind_plot = k_plot<5*10**-2
    plt.plot(k_plot[ind_plot],S_plot[ind_plot], c = colors[i], marker=mark[2], 
             label = str(int(dir_mean*180/np.pi))+' degrees' + 'L $<$ 500 and G $>=$ 2' )
    ind_3 = ind_dir & ind3
    k_plot = np.mean(np.array(k_H)[ind_3,:],axis=0)
    S_plot = np.mean(np.array(H_s_int)[ind_3,:],axis=0)
    ind_plot = k_plot<5*10**-2
    plt.plot(k_plot[ind_plot],S_plot[ind_plot], c = colors[i], marker=mark[3], 
             label = str(int(dir_mean*180/np.pi))+' degrees' + 'L $<$ 500 and G $<$ 2' )
    
    
#    plt.plot(np.mean(np.array(k_H)[ind,:],axis=0),np.mean(np.array(H)[ind,:],axis=0)+np.std(np.array(H)[ind,:],axis=0),'--')
#    plt.plot(np.mean(np.array(k_H)[ind,:],axis=0),np.mean(np.array(H)[ind,:],axis=0)-np.std(np.array(H)[ind,:],axis=0),'--')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k1')
    plt.ylabel('H')
plt.legend()

# In[Fitting]

def filter_H(param,args=()): 
    #param = [w,n]
    #args = (k)
    w,n,s= param  
    k_1 = args[0]
    return s/(1+(k_1*w)**n)

def cost(param,args=()):
    H = args[0](param,args=(args[1],))
    H_i = args[2] 
    #print(param,np.sum((np.log(H)-np.log(H_i))**2))
    return np.sum((np.log(H)-np.log(H_i))**2)

with open('H.pkl', 'rb') as V_t:
     k_H,H,sym = pickle.load(V_t)       
with open('H_s.pkl', 'rb') as V_t:
     k_H_s,H_s,sym_s = pickle.load(V_t) 

H_s_int = []
for i in range(len(H_s)):
    H_s_int.append(np.exp(sp.interpolate.interp1d(np.log(k_H_s[i]),
            np.log(H_s[i]))(np.log(k_H[i]))))     

param = []
param_init=[50,4,.6]
ind = k_H[i]<6*10**-2
for i in range(len(H_s_int)):
    res = sp.optimize.minimize(cost, param_init, args=((filter_H,k_H[i][ind],H_s_int[i][ind]),),method='Nelder-Mead')#'SLSQP',options={'ftol': 1e-10})#, bounds = bound)#,callback=callbackF, options={'disp': True})
    param.append(res.x)

plt.figure()
plt.plot(k_H[i],H_s_int[i])
plt.plot(k_H[i],filter_H(res.x,args=(k_H[i],)))
plt.xscale('log')
plt.yscale('log')


plt.figure()
plt.plot(k_H[i][ind],np.log(H_s_int[i][ind]))
plt.plot(k_H[i][ind],np.log(filter_H(res.x,args=(k_H[i][ind],))))


with open('simr.pkl', 'rb') as V_t:
     sym = pickle.load(V_t)      

ind0 = (np.array(sym)[:,3] >= 2 ) & (np.array(sym)[:,2] >= 500 )
ind1 = (np.array(sym)[:,3] < 2 ) & (np.array(sym)[:,2] >= 500 )
ind2 = (np.array(sym)[:,3] >= 2 ) & (np.array(sym)[:,2] < 500 )
ind3 = (np.array(sym)[:,3] < 2 ) & (np.array(sym)[:,2] < 500 )
 
plt.figure()
#for i, dir_mean in enumerate(Dir[:4]):
#    ind_dir = (np.array(sym)[:,0] == dir_mean*180/np.pi)
#    ind_0 = ind_dir & ind0

plt.hist(np.array(param)[:,0], bins=50)

plt.hist(np.array(param)[ind0,1], bins=50,
         label = 'L  $>=$ 500 and G $>=$ 2' )

plt.hist(np.array(param)[ind1,1], bins=50, 
         label = 'L $>=$ 500 and G $<$ 2' )

plt.hist(np.array(param)[ind2,1], bins=50, 
         label = 'L $<$ 500 and G $>=$ 2')


ind_dir = (np.array(sym)[:,0] == Dir[3]*180/np.pi)
ind_3 = ind3 = (np.array(sym)[:,3] < 2 ) & (np.array(sym)[:,2] < 500 )

ind3 = (np.array(sym)[:,3] < 5 ) & (np.array(sym)[:,2] <500)
plt.hist(np.array(param)[ind3,1], bins=50, 
         label = 'L $<$ 500 and G $<$ 2' )

