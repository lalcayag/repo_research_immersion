# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:27:46 2018

Module for 2D autocorrelation and spectra from horizontal wind field  
measurements. The structure expected is a triangulation from
scattered positions.

Autocorrelation is calculated in terms 

To do:
    
- Lanczos interpolaton on rectangular grid (usampling)

@author: lalc
"""
# In[Packages used]
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.tri import Triangulation,UniformTriRefiner,CubicTriInterpolator,LinearTriInterpolator,TriFinder,TriAnalyzer

from matplotlib import ticker    
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable  
                
from sklearn.neighbors import KDTree
from scipy.spatial import Delaunay

import time
import copy
# In[Autocorrelation for non-structured grid, brute force for non-interpolated wind field]  

def spatial_autocorr_sq(grid,U,V, mask_int = [], transform = True, transform_r = False, tri_calc = True, tri_del = [], refine=32,frac=.5, gamma = [], e_lim=[]):
    """
    Function to estimate autocorrelation in cartesian components of wind
    velocity, U and V. The 2D spatial autocorrelation is calculated in a 
    squared and structured grid, and represent the correlation of points
    displaced a distance tau and eta in x and y, respectively.

    Input:
    -----
        grid     - Grid of cartesian coordinates.
        
        U,V      - Arrays with cartesian components of wind speed.

        
    Output:
    ------
        r_u,r_v  - 2D arrays with autocorrelation function rho(tau,eta) 
                   for U and V, respectively.  
                   
    """  
    # Squared grid of spatial increments  
    
#######################################################################
 # Rotation of U, V
    U = U.astype(np.float32)
    V = V.astype(np.float32)
    t0 = time.process_time()
    t_0 = t0
    x = grid[0][0,:]
    y = grid[1][:,0]
    dx = np.diff(grid[0][0,:])[0]
    dy = np.diff(grid[1][:,0])[0]
    if transform:  
        if tri_calc:
            
            U, V, mask,mask_int, tri_del = field_rot(x, y, U, V, gamma = gamma, grid = grid,
                                            tri_calc = tri_calc)
        else:
            U, V, mask, _, _ = field_rot(x, y, U, V, gamma = gamma, grid = grid,
                                            tri_calc = tri_calc)       
        U[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, U[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
        U[~mask_int] = np.nan
        V[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, V[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
        V[~mask_int] = np.nan      
    
        U = np.reshape(U,grid[0].shape)
        V = np.reshape(V,grid[0].shape)

#######################################################################
    print('rotating:', time.process_time()-t0)
    t0 = time.process_time()
    _,_,U = shrink(grid,U)
    grd_shr_x,grd_shr_y,V = shrink(grid,V)
    grd_shr = (grd_shr_x,grd_shr_y)
    print('shrinking:', time.process_time()-t0)
    t0 = time.process_time()
    n, m = grd_shr[0].shape
    n_tau, n_eta = autocorr_grid(m,n,refine,frac) 
    tau = n_tau *np.min(np.diff(grd_shr[0][0,:]))
    eta = n_eta *np.min(np.diff(grd_shr[1][:,0]))   
    
    # De-meaning of U and V. The mean U and V in the whole scan is used
    U = U-np.nanmean(U)
    V = V-np.nanmean(V) 
    print('meshing:', time.process_time()-t0)
    t0 = time.process_time()
    # Autocorrelation is calculated just for non-empty scans 
    if len(U[~np.isnan(U)])>0:
        # autocorr() function over the grid tau and eta.
        ru = [autocorr_sq(U,t,e) for t, e in zip(n_tau[n_tau>=0].flatten(),n_eta[n_tau>=0].flatten())]
        print('autcorrelation u:', time.process_time()-t0)
        t0 = time.process_time()
        rv = [autocorr_sq(V,t,e) for t, e in zip(n_tau[n_tau>=0].flatten(),n_eta[n_tau>=0].flatten())]
        print('autcorrelation v:', time.process_time()-t0)
        t0 = time.process_time()
        r_uv = [crosscorr_sq(U,V,t,e) for t, e in zip(n_tau.flatten(),n_eta.flatten())]
        print('autcorrelation uv:', time.process_time()-t0)
        t0 = time.process_time()
    else:
        r_u = np.empty(len(tau.flatten()))
        r_u[:] = np.nan
        r_v = np.empty(len(tau.flatten()))
        r_v[:] = np.nan
        
    print('shape corr.', np.array(ru).shape)
    val_points = np.array(ru)[:,1]
    ru = np.array(ru)[:,0]
    rv = np.array(rv)[:,0]

    r_u = np.zeros(tau.shape)
    r_v = np.zeros(tau.shape)
    valid = np.zeros(tau.shape)
    
    r_u[n_tau>=0] = ru
    r_u[n_tau<0] = np.flip(r_u[n_tau>0])
    r_v[n_tau>=0] = rv
    r_v[n_tau<0] = np.flip(r_v[n_tau>0])
    valid[n_tau>=0] = val_points
    valid[n_tau<0] = np.flip(valid[n_tau>0])    
    r_uv = np.reshape(np.array(r_uv),tau.shape)
    valid,indicator,e = area_and_lenght_scale(r_u,valid,tau,eta,dx,dy,e_lim)
#    print(e.shape,tau[0,:].shape,eta[:,0].shape,r_u.shape)
    egrad = np.gradient(e,eta[:,0],tau[0,:])
    r_u = r_u/valid
    r_v = r_v/valid
    r_uv = r_uv/valid
    r_u[indicator==0] = np.nan
    r_v[indicator==0] = np.nan
    r_uv[indicator==0] = np.nan
    
    if transform_r:
        t0 = time.process_time() 
        S11 = np.cos(-gamma)
        S12 = np.sin(-gamma)
        T = np.array([[S11,S12], [-S12,S11]])
        S11 = np.cos(gamma)
        S12 = np.sin(gamma)
        tau_eta = np.array(np.c_[tau.flatten(),eta.flatten()]).T
        tau_eta = np.dot(T,tau_eta)       
        tau_prime = tau_eta[0,:]
        eta_prime = tau_eta[1,:]            
        r_u_prime = S11**2*r_u + S12**2*r_v + S11*S12*(r_uv+np.flip(r_uv))
        r_v_prime = S12**2*r_u + S11**2*r_v - S11*S12*(r_uv+np.flip(r_uv))
        r_uv_prime = -S11*S12*r_u + S11**2*r_uv - S12**2*np.flip(r_uv) + S11*S12*r_v
        mask_r = np.isnan(r_u_prime.flatten())
        r_u = sp.interpolate.griddata(np.c_[tau_prime,eta_prime][~mask_r],
              r_u_prime.flatten()[~mask_r], (tau.flatten(),eta.flatten()),
              method='cubic')  
        r_v = sp.interpolate.griddata(np.c_[tau_prime,eta_prime][~mask_r],
              r_v_prime.flatten()[~mask_r], (tau.flatten(),eta.flatten()),
              method='cubic')    
        r_uv = sp.interpolate.griddata(np.c_[tau_prime,eta_prime][~mask_r],
              r_uv_prime.flatten()[~mask_r], (tau.flatten(),eta.flatten()),
              method='cubic')
#        valid = sp.interpolate.griddata(np.c_[tau_prime,eta_prime],
#              valid.flatten(), (tau.flatten(),eta.flatten()),
#              method='cubic')
        
        print('rotating_r:', time.process_time()-t0)  
        r_u = np.reshape(r_u,tau.shape)
        r_v = np.reshape(r_v,tau.shape)
        r_uv = np.reshape(r_uv,tau.shape)
    
    _,_,r_u = shrink((tau,eta),r_u)
    _,_,r_v = shrink((tau,eta),r_v)
    tau,eta,r_uv = shrink((tau,eta),r_uv)
    r_u[tau<0] = np.flip(r_u[tau>0])
    r_v[tau<0] = np.flip(r_v[tau>0])
   
    print('total_time:', time.process_time()-t_0)
    return(tau,eta,r_u,r_v,r_uv,valid,indicator,e,egrad)
    
def shrink(grid,U):
    patch = ~np.isnan(U)
    ind_patch_x = np.sum(patch,axis=1) != 0
    ind_patch_y = np.sum(patch,axis=0) != 0
#    if np.sum(ind_patch_x) > np.sum(ind_patch_y):
#        ind_patch_y = ind_patch_x
#    elif np.sum(ind_patch_y) > np.sum(ind_patch_x):
#        ind_patch_x = ind_patch_y        
    n = np.sum(ind_patch_x)
    m = np.sum(ind_patch_y)          
    ind_patch_grd = np.meshgrid(ind_patch_y,ind_patch_x)
    ind_patch_grd = ind_patch_grd[0] & ind_patch_grd[1]
    U = np.reshape(U[ind_patch_grd],(n,m))
    grid_x = np.reshape(grid[0][ind_patch_grd],(n,m))
    grid_y = np.reshape(grid[1][ind_patch_grd],(n,m))
    return (grid_x,grid_y,U)    
    
def autocorr_sq(U,n_tau,n_eta):
    
    if (n_tau!=0) & (n_eta!=0):
        if (n_tau>0) & (n_eta>0): #ok
            U_del = U[n_eta:,:-n_tau]
            U = U[:-n_eta,n_tau:]                        
        if (n_tau<0) & (n_eta>0): #ok
            U_del = U[n_eta:,-n_tau:]
            U = U[:-n_eta,:n_tau]                    
        if (n_tau>0) & (n_eta<0): #ok
            U_del = U[:n_eta,:-n_tau]
            U = U[-n_eta:,n_tau:]
        if (n_tau<0) & (n_eta<0): #ok
            U_del = U[:n_eta,-n_tau:] 
            U = U[-n_eta:,:n_tau]                       
    if (n_tau==0) & (n_eta!=0):
        if n_eta>0:
            U_del = U[n_eta:,:]
            U = U[:-n_eta,:]                    
        if n_eta<0:
            U_del = U[:n_eta,:] 
            U = U[-n_eta:,:]            
    if (n_tau!=0) & (n_eta==0):
        if n_tau>0:
            U_del = U[:,:-n_tau]
            U = U[:,n_tau:]             
        if n_tau<0:
            U_del = U[:,-n_tau:] 
            U = U[:,:n_tau]           
    if (n_tau==0) & (n_eta==0):
        U_del = U
    ind = ~(np.isnan(U.flatten()) | np.isnan(U_del.flatten()))
    
    return (np.sum(U.flatten()[ind]*U_del.flatten()[ind]),np.sum(ind))

def crosscorr_sq(U,V,n_tau,n_eta):
    if (n_tau!=0) & (n_eta!=0):
        if (n_tau>0) & (n_eta>0): #ok
            U_del = V[n_eta:,:-n_tau]
            U = U[:-n_eta,n_tau:]                        
        if (n_tau<0) & (n_eta>0): #ok
            U_del = V[n_eta:,-n_tau:]
            U = U[:-n_eta,:n_tau]                    
        if (n_tau>0) & (n_eta<0): #ok
            U_del = V[:n_eta,:-n_tau]
            U = U[-n_eta:,n_tau:]
        if (n_tau<0) & (n_eta<0): #ok
            U_del = V[:n_eta,-n_tau:] 
            U = U[-n_eta:,:n_tau]                       
    if (n_tau==0) & (n_eta!=0):
        if n_eta>0:
            U_del = V[n_eta:,:]
            U = U[:-n_eta,:]                    
        if n_eta<0:
            U_del = V[:n_eta,:] 
            U = U[-n_eta:,:]            
    if (n_tau!=0) & (n_eta==0):
        if n_tau>0:
            U_del = V[:,:-n_tau]
            U = U[:,n_tau:]             
        if n_tau<0:
            U_del = V[:,-n_tau:] 
            U = U[:,:n_tau]           
    if (n_tau==0) & (n_eta==0):
        U_del = V
    ind = ~(np.isnan(U.flatten()) | np.isnan(U_del.flatten()))
    return np.sum(U.flatten()[ind]*U_del.flatten()[ind])

def autocorr_interp_sq(r, eta, tau, N = [], tau_lin = [], eta_lin = []):
    if (len(eta_lin) == 0) | (len(eta_lin) == 0):
        if len(N) == 0:
            N = 2**(int(np.ceil(np.log(np.max([tau.shape[1],eta.shape[0]]))/np.log(2)))+3)
        tau_lin = np.linspace(np.min(tau.flatten()),np.max(tau.flatten()),N)
        eta_lin = np.linspace(np.min(eta.flatten()),np.max(eta.flatten()),N)
        tau_lin, eta_lin = np.meshgrid(tau_lin,eta_lin)
    ind = ~np.isnan(r.flatten())
    tri_tau = Delaunay(np.c_[tau.flatten()[ind],eta.flatten()[ind]])   
    r_int = sp.interpolate.CloughTocher2DInterpolator(tri_tau, r.flatten()[ind])(np.c_[tau_lin.flatten(),eta_lin.flatten()])
    return (tau_lin,eta_lin,np.reshape(r_int,tau_lin.shape))

def shrink_domain(grid,U,V):
        patch = ~np.isnan(U)
        ind_patch_x = np.sum(patch,axis=1) != 0
        ind_patch_y = np.sum(patch,axis=0) != 0
        if np.sum(ind_patch_x) > np.sum(ind_patch_y):
            ind_patch_y = ind_patch_x
        elif np.sum(ind_patch_y) > np.sum(ind_patch_x):
            ind_patch_x = ind_patch_y        
        n = np.sum(ind_patch_x)
        m = np.sum(ind_patch_y)          
        ind_patch_grd = np.meshgrid(ind_patch_y,ind_patch_x)
        ind_patch_grd = ind_patch_grd[0] & ind_patch_grd[1]
        U = np.reshape(U[ind_patch_grd],(n,m))
        V = np.reshape(V[ind_patch_grd],(n,m)) 
        grid_x = np.reshape(grid[0][ind_patch_grd],(n,m))
        grid_y = np.reshape(grid[1][ind_patch_grd],(n,m))
        return ((grid_x,grid_y),U,V)
    
def autocorr_grid(n,m,refine,frac):
    n_tau_max = n
    n_eta_max = m   
    n_tau_next = n_tau_max#int(2**np.floor(np.log(n_tau_max)/np.log(2))) 
    n_eta_next = n_eta_max#int(2**np.floor(np.log(n_eta_max)/np.log(2)))    
    n_tau_exp = np.unique(np.round(np.exp((np.linspace(0,np.log(n_tau_next),refine))).astype(int)))
    n_tau_exp = n_tau_exp[:int(frac*len(n_tau_exp))+1]
    d_tau_exp = np.diff(n_tau_exp)[-1]
    n_tau_lin = np.arange(n_tau_exp[-1],n_tau_next,d_tau_exp)[1:]
    n_tau = np.r_[n_tau_exp,n_tau_lin]
    n_tau = np.r_[-np.flip(n_tau),0,n_tau]   
    n_eta_exp = np.unique(np.round(np.exp((np.linspace(0,np.log(n_eta_next),refine))).astype(int)))
    n_eta_exp = n_eta_exp[:int(frac*len(n_eta_exp))+1]
    d_eta_exp = np.diff(n_eta_exp)[-1]
    n_eta_lin = np.arange(n_eta_exp[-1],n_eta_next,d_eta_exp)[1:]
    n_eta = np.r_[n_eta_exp,n_eta_lin]
    n_eta = np.r_[-np.flip(n_eta),0,n_eta]     
    n_tau,n_eta = np.meshgrid(n_tau,n_eta)
    return (n_tau,n_eta)

def area_and_lenght_scale(r_u,valid,tau,eta,dx,dy,e_lim):
    zero_crossings_x = np.where(np.diff(np.sign(np.nanmean(r_u,axis=0))))[0]
    zero_crossings_y = np.where(np.diff(np.sign(np.nanmean(r_u,axis=1))))[0]
    if len(zero_crossings_x) == 0:
        zero_crossings_x = 0
        tau_zero = tau[0,:][zero_crossings_x]
    else:
        tau_zero = tau[0,:][zero_crossings_x][np.argsort(np.abs(tau[0,:][zero_crossings_x]-tau[0,:][tau[0,:]==0]))[1]]    
    if len(zero_crossings_y) == 0:
        zero_crossings_y = 0
        eta_zero = eta[:,0][zero_crossings_y]
    else:
        eta_zero = eta[:,0][zero_crossings_y][np.argsort(np.abs(eta[:,0][zero_crossings_y]-eta[:,0][eta[:,0]==0]))[1]]
    valid_zero = valid.copy()
    indicator = np.ones(valid.shape)
    valid_zero[valid==0] = np.nan
    indicator[valid==0] = 0
    ucorr_zero = r_u.copy()
    ucorr_zero[valid==0] = np.nan
    u_corr_mean_x = np.nanmean(r_u,axis=0)/np.nanmean(valid_zero,axis=0)
    u_corr_mean_y = np.nanmean(r_u,axis=1)/np.nanmean(valid_zero,axis=1)
    u_corr_mean_x = u_corr_mean_x/np.nanmax(u_corr_mean_x)
    u_corr_mean_y = u_corr_mean_y/np.nanmax(u_corr_mean_y)
    ind_tau = np.abs(tau[0,:]) > np.abs(tau_zero)
    ind_eta = np.abs(eta[:,0]) > np.abs(eta_zero)
    ind_e = np.meshgrid(~ind_tau,~ind_eta)
    ind_e = ind_e[0] & ind_e[1]
    u_corr_mean_x[ind_tau] = 0
    u_corr_mean_y[ind_eta] = 0
    u_corr_mean_x[np.isnan(u_corr_mean_x)] = 0
    u_corr_mean_y[np.isnan(u_corr_mean_y)] = 0   
    Lx = np.trapz(u_corr_mean_x,tau[0,:])/2
    Ly = np.trapz(u_corr_mean_y,eta[:,0])/2
    e = np.sqrt(2*Lx*Ly/(valid*dx*dy))
    elim = np.min([np.nanmax(e[ind_e])*1.1,e_lim])
    print(elim, np.nanmax(e[ind_e]),np.nanmin(e))
    valid_zero[e>elim] = np.nanmax(valid_zero[e>elim])
    return (valid_zero,indicator,e)

def integral_lenght_scale(r_u,tau,eta):
    zero_axis_eta = np.where(eta==0)[0]
    zero_axis_tau = np.where(tau==0)[0]
    rx = np.squeeze(r_u[zero_axis_eta,:])
    ry = np.squeeze(r_u[:,zero_axis_tau])
    zero_crossings_x = np.where(np.diff(np.sign(rx)))[0]
    zero_crossings_y = np.where(np.diff(np.sign(ry)))[0]
    if len(zero_crossings_x) == 0:
        zero_crossings_x = 0
        tau_zero = tau[zero_crossings_x]
    else:
        tau_zero = tau[zero_crossings_x][np.argsort(np.abs(tau[zero_crossings_x]-tau[tau==0]))[1]]    
    if len(zero_crossings_y) == 0:
        zero_crossings_y = 0
        eta_zero = eta[zero_crossings_y]
    else:
        eta_zero = eta[zero_crossings_y][np.argsort(np.abs(eta[zero_crossings_y]-eta[eta==0]))[1]]

    ind_tau = np.abs(tau) > np.abs(tau_zero)
    ind_eta = np.abs(eta) > np.abs(eta_zero) 
    Lx = np.trapz(rx[~ind_tau]/np.max(rx),tau[~ind_tau])/2
    Ly = np.trapz(ry[~ind_eta]/np.max(ry),eta[~ind_eta])/2
    return (Lx,Ly)
                
# In[Autocorrelation from triangulation]
def spatial_autocorr(tri,U,V,N,alpha):
    """
    Function to estimate autocorrelation in cartesian components of wind
    velocity, U and V. The 2D spatial autocorrelation is calculated in a 
    squared and structured grid, and represent the correlation of points
    displaced a distance tau and eta in x and y, respectively.

    Input:
    -----
        tri      - Delaunay triangulation object of unstructured grid.
        
        U,V      - Arrays with cartesian components of wind speed.
        
        N        - Number of points in the autocorrelation's squared grid.
        
        alpha    - Fraction of the spatial domain that will act as the limit 
                   for tau and eta increments. 
        
    Output:
    ------
        r_u,r_v  - 2D arrays with autocorrelation function rho(tau,eta) 
                   for U and V, respectively.  
                   
    """
    
    # Squared grid of spatial increments
    tau = np.linspace(-alpha*(np.max(tri.x)-np.min(tri.x)),alpha*(np.max(tri.x)-np.min(tri.x)),N)
    eta = np.linspace(-alpha*(np.max(tri.y)-np.min(tri.y)),alpha*(np.max(tri.y)-np.min(tri.y)),N)
    tau,eta = np.meshgrid(tau,eta)
    # De-meaning of U and V. The mean U and V in the whole scan is used
    U = U-np.nanmean(U)
    V = V-np.nanmean(V)
    # Interpolator object to estimate U and V fields when translated by
    # (tau,eta)
    U_int = LinearTriInterpolator(tri, U)
    V_int = LinearTriInterpolator(tri, V)
    # Autocorrelation is calculated just for non-empty scans 
    if len(U[~np.isnan(U)])>0:
        # autocorr() function over the grid tau and eta.
        r_u = [autocorr(tri,U,U_int,t,e) for t, e in zip(tau.flatten(),eta.flatten())]
        r_v = [autocorr(tri,V,V_int,t,e) for t, e in zip(tau.flatten(),eta.flatten())]
    else:
        r_u = np.empty(len(tau.flatten()))
        r_u[:] = np.nan
        r_v = np.empty(len(tau.flatten()))
        r_v[:] = np.nan
    
    return(r_u,r_v)

def autocorr(tri,U,Uint,tau,eta):
    """
    Function to estimate autocorrelation from a single increment in cartesian
    coordinates(tau, eta)
    Input:
    -----
        tri      - Delaunay triangulation object of unstructured grid.
        
        U        - Arrays with a cartesian component wind speed.
        
        U_int    - Linear interpolator object.
        
        tau      - Increment in x coordinate. 
        
        eta      - Increment in y coordinate.
        
    Output:
    ------
        r        - Autocorrelation function value.  
                   
    """ 
    # Only un-structured grid with valid wind speed
    ind = ~np.isnan(U)
    # Interpolation of U for a translation of the grid by (tau,eta)
    U_delta = Uint(tri.x[ind]+tau,tri.y[ind]+eta)
    # Autocorrelation on valid data in the original unstructured grid and the
    # displaced one.
    if len(U_delta.data[~U_delta.mask]) == 0:
        r = np.nan
    else:
        # Autocorrelation is the off-diagonal value of the correlation matrix.
        r = np.corrcoef(U_delta.data[~U_delta.mask],U[ind][~U_delta.mask],
                        rowvar=False)[0,1]
    return r

# In[Just spectra]
def spectra_fft(grid,U,V,UV,K=0,inv=False):
    """
    Input:
    -----                
        grid              - Squared, structured grid to apply FFT.
        
        U, V              - Arrays with cartesian components of wind speed.
        
    Output:
    ------
        k1,k2,Suu,Svv,Suv - 2D arrays with wavenumber and power spectra
                            for U, V and UV, respectively.               
    """
    grid_0 = grid[0].copy()
    grid_1 = grid[1].copy()   
    grid_0 = grid_0[:-1,:-1]
    grid_1 = grid_1[:-1,:-1] 
    U = U[:-1,:-1]
    V = V[:-1,:-1]
    UV = UV[:-1,:-1]    
    dx = np.max(np.diff(grid_0.flatten()))
    dy = np.max(np.diff(grid_1.flatten()))  
    n = grid_0.shape[0]
    m = grid_1.shape[1] 
    #zero padding
    Uaux = np.zeros((n*2**K,m*2**K))
    Uaux[:n,:m] = U
    Vaux = np.zeros((n*2**K,m*2**K))
    Vaux[:n,:m] = V
    UVaux = np.zeros((n*2**K,m*2**K))
    UVaux[:n,:m] = UV
    
    if inv:
        fftU  = np.fft.fftshift(np.fft.ifft2(Uaux))
        fftV  = np.fft.fftshift(np.fft.ifft2(Vaux))
        fftUV = np.fft.fftshift(np.fft.ifft2(UVaux))
        k1 = np.fft.fftshift((np.fft.fftfreq(n*2**K, d=dx)))
        k2 = np.fft.fftshift((np.fft.fftfreq(m*2**K, d=dy)))
    else:
        fftU  = np.fft.fftshift(np.fft.fft2(Uaux))*dx*dy/(2*np.pi)**2
        fftV  = np.fft.fftshift(np.fft.fft2(Vaux))*dx*dy/(2*np.pi)**2
        fftUV = np.fft.fftshift(np.fft.fft2(UVaux))*dx*dy/(2*np.pi)**2
        k1 = np.fft.fftshift((np.fft.fftfreq(m*2**K, d=dx/2/np.pi)))
        k2 = np.fft.fftshift((np.fft.fftfreq(n*2**K, d=dy/2/np.pi))) 
    return(k1,k2,np.abs(fftU),np.abs(fftV),fftUV)

# In[Spectra smoothing]
def spec_smooth(S,k1,k2,k1_bin_c,k2_bin_c,bins):
    
    k1_min = np.ceil(np.log10(np.min(k1_bin_c[k1_bin_c>0])))
    k1_max = np.floor(np.log10(np.max(k1_bin_c[k1_bin_c>0])))
    k2_min = np.ceil(np.log10(np.min(k2_bin_c[k2_bin_c>0])))
    k2_max = np.floor(np.log10(np.max(k2_bin_c[k2_bin_c>0])))
    k1_range = np.r_[np.arange(k1_min,k1_max),k1_max]
    k2_range = np.r_[np.arange(k2_min,k2_max),k2_max]
    print(k1_max)
    k1_l = []
    k2_l = []
    for i in range(len(k1_range)):
        for j in range(bins):
            k1_l.append(10**(k1_range[i]+j/bins))

    for i in range(len(k2_range)):
        for j in range(bins):
            k2_l.append(10**(k2_range[i]+j/bins))    
    k1_l = np.array(k1_l)
    k1_l = k1_l[k1_l<=np.max(k1_bin_c[k1_bin_c>0])]
    k2_l = np.array(k2_l)
    k2_l = k2_l[k2_l<=np.max(k2_bin_c[k2_bin_c>0])]
    k1_range = np.r_[-np.max(k1_bin_c[k1_bin_c>0]),-np.flip(k1_l),0,k1_l,np.max(k1_bin_c[k1_bin_c>0])]
    k2_range = np.r_[-np.max(k2_bin_c[k1_bin_c>0]),-np.flip(k2_l),0,k2_l,np.max(k1_bin_c[k2_bin_c>0])]
    k1_bin_c,k2_bin_c = np.meshgrid(k1_range,k2_range)
    print(k1_range)
 
    S = np.reshape(S,(k2.shape[0],k1.shape[0]))
    k1,k2 = np.meshgrid(k1,k2)
    
#    indk1 = np.abs(k1)<=np.max(np.abs(k1_bin_c)) + np.diff(k1_bin_c)[0]/2
#    indk2 = np.abs(k2)<=np.max(np.abs(k2_bin_c)) + np.diff(k2_bin_c)[0]/2
#    
#    indk = indk1 & indk2   
#    S = S[indk]
#    k1 = k1[indk]
#    k2 = k2[indk]  
#    print(k1.shape,k2.shape,S.shape)
    
#    k1_bin_c,k2_bin_c = np.meshgrid(k1_bin_c,k2_bin_c)  
    

    
    tree_k_c = KDTree(np.c_[k1_bin_c.flatten(),k2_bin_c.flatten()],metric='manhattan')
    
    ind = tree_k_c.query(np.c_[k1.flatten(),k2.flatten()],return_distance=False)
    
    ind = np.squeeze(ind)
    
    print(ind.shape,np.unique(ind).shape,k1_bin_c.flatten().shape)  
    
    ind_sorted = np.argsort(ind)
    
    Si_sorted = S.flatten()[ind_sorted] 
    
    delta_bin = ind[ind_sorted][1:] - ind[ind_sorted][:-1]
    bin_ind = np.r_[0,np.where(delta_bin)[0],ind[ind_sorted][-1]] # location of changes in bin
    print(bin_ind.shape)
    nr = bin_ind[1:] - bin_ind[:-1]  # number of elements per bin
    print(nr.shape)
    csSim = np.cumsum(np.log(Si_sorted), dtype=float)
    tbin = csSim[bin_ind[1:]] - csSim[bin_ind[:-1]]
    print(tbin.shape)
    Si_bin = np.exp(sp.interpolate.griddata(np.c_[k1_bin_c.flatten(),k2_bin_c.flatten()][np.unique(ind)],
              tbin/nr, (k1_bin_c.flatten(),k2_bin_c.flatten()),
              method='nearest'))
    plt.figure()
    plt.contourf(k1_bin_c,k2_bin_c,np.log(np.reshape(Si_bin,k1_bin_c.shape)),np.linspace(-30,15,10),cmap='jet')
    plt.colorbar()
    print(Si_bin.shape,k1_bin_c.flatten().shape)
    return ((k1_bin_c,k2_bin_c),np.reshape(Si_bin,k1_bin_c.shape))
    
# In[]
def field_rot(x, y, U, V, gamma = None, grid = [], tri_calc = False, tri_del = []):
    
    if gamma == None:
        
        U_mean = np.nanmean(U.flatten())
        V_mean = np.nanmean(V.flatten())
        # Wind direction
        gamma = np.arctan2(V_mean,U_mean)
    # Components in matrix of coefficients
    S11 = np.cos(gamma)
    S12 = np.sin(gamma)
    R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
    
    vel = np.array(np.c_[U.flatten(),V.flatten()]).T
    vel = np.dot(R[:-1,:-1],vel)
    U = vel[0,:]
    V = vel[1,:]
    mask = ~np.isnan(U)
    mask_int = []
  
    if tri_calc:
        if not grid:
            grid = np.meshgrid(x,y)       
        xtrans = 0
        ytrans = y[0]/2
        T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
        T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
        T = np.dot(np.dot(T1,R),T2)
        Xx = np.array(np.c_[grid[0].flatten(),grid[1].flatten(),np.ones(len(grid[0].flatten()))]).T
        Xx = np.dot(T,Xx)   
        tri_del = Delaunay(np.c_[Xx[0,:][mask],Xx[1,:][mask]])
        mask_int = ~(tri_del.find_simplex(np.c_[grid[0].flatten(),grid[1].flatten()]) == -1)  
        
    return (U, V, mask, mask_int, tri_del)    
     
# In[]
def spatial_spec_sq(x,y,U_in,V_in, tri_del = [], mask_int = [], tri_calc = True, transform = False, shrink = True, ring=False, plot = False, one_dim = False, bins=20,K=0):

    U = U_in.copy()
    V = V_in.copy()
    n_p = np.sum(np.isnan(U))
    grid = np.meshgrid(x,y)
    n, m = grid[0].shape
    dx = np.min(np.abs(np.diff(x)))
    dy = np.min(np.abs(np.diff(y)))
    a_eff = (n*m-n_p)*dx*dy
#    a_eff = n*m*dx*dy
    
#######################################################################
 # Rotation of U, V
    if transform:
        
        if tri_calc:
            
            U, V, mask,mask_int, tri_del = field_rot(x, y, U, V, gamma = None, grid = grid,
                                            tri_calc = tri_calc)
        else:
            U, V, mask, _, _ = field_rot(x, y, U, V, gamma = None, grid = grid,
                                            tri_calc = tri_calc)
         
        print('reducing da domain')  
            
        print('interp U ',type(tri_del))
        U[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, U[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
        U[~mask_int] = np.nan
        print('interp V ',type(tri_del))
        V[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, V[mask])(np.c_[grid[0].flatten(),grid[1].flatten()][mask_int])
        V[~mask_int] = np.nan
        print('interp ends')
 
        n_p = np.sum(np.isnan(U))
        a_eff = (n*m-n_p)*dx*dy        
#        a_eff = n*m*dx*dy
        U = np.reshape(U,grid[0].shape)
        V = np.reshape(V,grid[0].shape)

####################################################################### 
    # Shrink and square
    if shrink:
        patch = ~np.isnan(U)
        ind_patch_x = np.sum(patch,axis=1) != 0
        ind_patch_y = np.sum(patch,axis=0) != 0
        if np.sum(ind_patch_x) > np.sum(ind_patch_y):
            ind_patch_y = ind_patch_x
        elif np.sum(ind_patch_y) > np.sum(ind_patch_x):
            ind_patch_x = ind_patch_y        
        n = np.sum(ind_patch_x)
        m = np.sum(ind_patch_y)          
        ind_patch_grd = np.meshgrid(ind_patch_y,ind_patch_x)
        ind_patch_grd = ind_patch_grd[0] & ind_patch_grd[1]
        U = np.reshape(U[ind_patch_grd],(n,m))
        V = np.reshape(V[ind_patch_grd],(n,m))
        grid[0] = np.reshape(grid[0][ind_patch_grd],(n,m))
        grid[1] = np.reshape(grid[1][ind_patch_grd],(n,m))        
     
    k1_int = np.fft.fftshift((np.fft.fftfreq(m*2**K, d=dx/2/np.pi)))
    k2_int = np.fft.fftshift((np.fft.fftfreq(n*2**K, d=dy/2/np.pi))) 
    k_int_grd = np.meshgrid(k1_int,k2_int)

    U_mean = np.nanmean(U.flatten())
    V_mean = np.nanmean(V.flatten())
    
    U_t = U-U_mean
    V_t = V-V_mean
    
    U_t[np.isnan(U_t)] = 0.0
    V_t[np.isnan(V_t)] = 0.0
    
    Uaux = np.zeros((n*2**K,m*2**K))
    Uaux[:n,:m] = U_t
    Vaux = np.zeros((n*2**K,m*2**K))
    Vaux[:n,:m] = V_t
    fftU  = np.fft.fftshift(np.fft.fft2(Uaux))*dx*dy/2/np.pi
    fftV  = np.fft.fftshift(np.fft.fft2(Vaux))*dx*dy/2/np.pi
#    fftUV = np.fft.fftshift(np.fft.fft2(UVaux))*dx*dy

    # Spectra
#    fftU = np.fft.fft2(U_t)
#    fftV = np.fft.fft2(V_t)
#    fftU  = np.fft.fftshift(fftU)
#    fftV  = np.fft.fftshift(fftV) 
    fftUV = fftU*np.conj(fftV) 
    
    Suu_int = np.abs(fftU)**2/a_eff
    Svv_int = np.abs(fftV)**2/a_eff
    Suv_int = np.real(fftUV)/a_eff

#####################################################################################################
    if plot:
        
        plot_log2D(k_int_grd, Suu_int, label_S = "$\log_{10}{Suu}$", C = 10**-4)
        plot_log2D(k_int_grd, Svv_int, label_S = "$\log_{10}{Svv}$", C = 10**-4)
        plot_log2D(k_int_grd, Suv_int, label_S = "$\log_{10}{Suv}$", C = 10**-4)
              
########################################################################################################
    
    if ring:
        Su=spectra_average(Suu_int,(k1_int, k2_int),bins,angle_bin = 30,stat=False)
        #ku = Su.k        
        Suu_int = Su.S    
        Sv=spectra_average(Svv_int,(k1_int, k2_int),bins,angle_bin = 30,stat=False)
        #kv = Sv.k
        Svv_int = Sv.S
        Suv=spectra_average(Suv_int,(k1_int, k2_int),bins,angle_bin = 30,stat=False)
        #kuv = Suv.k
        Suv_int = Suv.S
    if one_dim:
        print('yes')
        Suv_int = sp.integrate.simps(Suv_int.T,k2_int,axis=1)[k1_int>0]
        Suu_int = sp.integrate.simps(Suu_int.T,k2_int,axis=1)[k1_int>0]
        Svv_int = sp.integrate.simps(Svv_int.T,k2_int,axis=1)[k1_int>0]
        return (k1_int[k1_int>0],k2_int[k2_int>0],Suu_int,Svv_int,Suv_int) 
    else:
        return (k1_int, k2_int,Suu_int,Svv_int,Suv_int) 

# In[Autocorrelation for non-structured grid, using FFT for interpolated
#                                                         (or not) wind field] 
    
def spatial_autocorr_fft(tri,U,V,N_grid=512,auto=False,transform = False, tree = None, interp = 'Lanczos'):
    """
    Function to estimate autocorrelation from a single increment in cartesian
    coordinates(tau, eta)
    Input:
    -----
        tri      - Delaunay triangulation object of unstructured grid.
                   
        N_grid     - Squared, structured grid resolution to apply FFT.
        
        U, V     - Arrays with cartesian components of wind speed.
        
    Output:
    ------
        r_u,r_v  - 2D arrays with autocorrelation function rho(tau,eta) 
                   for U and V, respectively.               
    """   
    if transform:
        U_mean = avetriangles(np.c_[tri.x,tri.y], U, tri)
        V_mean = avetriangles(np.c_[tri.x,tri.y], V, tri)
        # Wind direction
        gamma = np.arctan2(V_mean,U_mean)
        # Components in matrix of coefficients
        S11 = np.cos(gamma)
        S12 = np.sin(gamma)
        T = np.array([[S11,S12], [-S12,S11]])
        vel = np.array(np.c_[U,V]).T
        vel = np.dot(T,vel)
        X = np.array(np.c_[tri.x,tri.y]).T
        X = np.dot(T,X)
        U = vel[0,:]
        V = vel[1,:]
        tri = Triangulation(X[0,:],X[1,:])
        mask=TriAnalyzer(tri).get_flat_tri_mask(.05)
        tri=Triangulation(tri.x,tri.y,triangles=tri.triangles[~mask])
        U_mean = avetriangles(np.c_[tri.x,tri.y], U, tri)
        V_mean = avetriangles(np.c_[tri.x,tri.y], V, tri)
    else:
        # Demeaning
        U_mean = avetriangles(np.c_[tri.x,tri.y], U, tri)
        V_mean = avetriangles(np.c_[tri.x,tri.y], V, tri)
        
    grid = np.meshgrid(np.linspace(np.min(tri.x),
           np.max(tri.x),N_grid),np.linspace(np.min(tri.y),
                 np.max(tri.y),N_grid))   
    
    U = U-U_mean
    V = V-V_mean
    
    # Interpolated values of wind field to a squared structured grid
                   
    if interp == 'cubic':     
        U_int= CubicTriInterpolator(tri, U)(grid[0].flatten(),grid[1].flatten()).data
        V_int= CubicTriInterpolator(tri, V)(grid[0].flatten(),grid[1].flatten()).data 
        U_int = np.reshape(U_int,grid[0].shape)
        V_int = np.reshape(V_int,grid[0].shape)
    else:
#        U_int= lanczos_int_sq(grid,tree,U)
#        V_int= lanczos_int_sq(grid,tree,V) 
        U_int= LinearTriInterpolator(tri, U)(grid[0].flatten(),grid[1].flatten()).data
        V_int= LinearTriInterpolator(tri, V)(grid[0].flatten(),grid[1].flatten()).data 
        U_int = np.reshape(U_int,grid[0].shape)
        V_int = np.reshape(V_int,grid[0].shape)
             
    #zero padding
    U_int[np.isnan(U_int)] = 0.0
    V_int[np.isnan(V_int)] = 0.0
    fftU = np.fft.fft2(U_int)
    fftV = np.fft.fft2(V_int)
    if auto:

        # Autocorrelation
        r_u = np.real(np.fft.fftshift(np.fft.ifft2(np.absolute(fftU)**2)))/len(U_int.flatten())
        r_v = np.real(np.fft.fftshift(np.fft.ifft2(np.absolute(fftV)**2)))/len(U_int.flatten())
        r_uv = np.real(np.fft.fftshift(np.fft.ifft2(np.real(fftU*np.conj(fftV)))))/len(U_int.flatten())
    
    dx = np.max(np.diff(grid[0].flatten()))
    dy = np.max(np.diff(grid[1].flatten()))
    
    n = grid[0].shape[0]
    m = grid[1].shape[0]   
    # Spectra
    fftU  = np.fft.fftshift(fftU)
    fftV  = np.fft.fftshift(fftV) 
    fftUV = fftU*np.conj(fftV) 
    Suu = 2*(np.abs(fftU)**2)*(dx*dy)/(n*m)
    Svv = 2*(np.abs(fftV)**2)*(dx*dy)/(n*m)
    Suv = 2*np.real(fftUV)*(dx*dy)/(n*m)
    k1 = np.fft.fftshift((np.fft.fftfreq(n, d=dx)))
    k2 = np.fft.fftshift((np.fft.fftfreq(m, d=dy)))
    if auto:
        return(r_u,r_v,r_uv,Suu,Svv,Suv,k1,k2)
    else:
        return(Suu,Svv,Suv,k1,k2)
 
# In[Ring average of spectra]
def spectra_average0(S_image,k,bins,angle_bin = 30,stat=False,log_space = True):
    """
    S_r = spectra_average(S_image,k,bins)
    
    A function to reduce 2D Spectra to a radial cross-section.
    
    INPUT:
    ------
        S_image   - 2D Spectra array.
        
        k         - Tuple containing (k1_max,k2_max), wavenumber axis
                    limits
        bins      - Number of bins per decade.
        
        angle_bin - Sectors to determine spectra alignment
        
        stat      - Bin statistics output
        
     OUTPUT:
     -------
      S_r - a data structure containing the following
                   statistics, computed across each annulus:
          .k      - horizontal wavenumber k**2 = k1**2 + k2**2
          .S      - mean of the Spectra in the annulus
          .std    - std. dev. of S in the annulus
          .median - median value in the annulus
          .max    - maximum value in the annulus
          .min    - minimum value in the annulus
          .numel  - number of elements in the annulus
    """
#    import numpy as np

    class Spectra_r:
        """Empty object container.
        """
        def __init__(self): 
            self.S = None
    #---------------------
    # Set up input parameters
    #---------------------
    S_image = np.array(S_image)
    npix, npiy = S_image.shape       
        
    k1 = k[0]
    k2 = k[1]
    k1, k2 = np.meshgrid(k1,k2)
    # Polar coordiantes (complex phase space)
    r = np.absolute(k1+1j*k2)

    Si_sorted = S_image.flatten()
    if log_space:
        r_log10 = np.log10(r.flatten())
        decades = len(np.unique(r_log10.astype(int)))    
        bin_tot = decades*bins
        r_bin10 = np.linspace(np.min(r_log10.astype(int))-1,np.max(r_log10.astype(int)),bin_tot+1)
    else:
        r_log10 = r.flatten()
        decades = len(np.unique(r_log10.astype(int)))    
        bin_tot = bins
        r_bin10 = np.linspace(np.min(r_log10),np.max(r_log10),bin_tot+1)
    
    S_ring = np.zeros(Si_sorted.shape)#tbin/nr

                
    for i in range(len(r_bin10)-1):
        ind0 = (r_log10>=r_bin10[i]) & (r_log10<r_bin10[i+1])
        S_ring[ind0] = np.sum(S_image.flatten()[ind0])/np.sum(ind0)   
    
    S_r = Spectra_r()
    S_r.S = np.reshape(S_ring,S_image.shape)

    if stat==True:
        S_stat = np.array([[np.std(Si_sorted[r_n_bin==r]), 
                            np.median(Si_sorted[r_n_bin==r]),
                            np.max(Si_sorted[r_n_bin==r]),
                            np.min(Si_sorted[r_n_bin==r])]
                            for r in np.unique(r_n_bin)])        
        S_r.std = S_stat[:,0]
        S_r.median = S_stat[:,1]
        S_r.max = S_stat[:,2]
        S_r.min = S_stat[:,3]
    
    return S_r

def spectra_average(S_image,k,bins,angle_bin = 30,stat=False):
    """
    S_r = spectra_average(S_image,k,bins)
    
    A function to reduce 2D Spectra to a radial cross-section.
    
    INPUT:
    ------
        S_image   - 2D Spectra array.
        
        k         - Tuple containing (k1_max,k2_max), wavenumber axis
                    limits
        bins      - Number of bins per decade.
        
        angle_bin - Sectors to determine spectra alignment
        
        stat      - Bin statistics output
        
     OUTPUT:
     -------
      S_r - a data structure containing the following
                   statistics, computed across each annulus:
          .k      - horizontal wavenumber k**2 = k1**2 + k2**2
          .S      - mean of the Spectra in the annulus
          .std    - std. dev. of S in the annulus
          .median - median value in the annulus
          .max    - maximum value in the annulus
          .min    - minimum value in the annulus
          .numel  - number of elements in the annulus
    """
#    import numpy as np

    class Spectra_r:
        """Empty object container.
        """
        def __init__(self): 
            self.S = None
            #self.std = None
            #self.median = None
            #self.numel = None
            #self.max = None
            #self.min = None
            #self.k1 = None
            #self.k2 = None   
    #---------------------
    # Set up input parameters
    #---------------------
    S_image = np.array(S_image)
    npix, npiy = S_image.shape       
        
    k1 = k[0]#*np.linspace(-1,1,npix)
    k2 = k[1]#*np.linspace(-1,1,npiy)
    k1, k2 = np.meshgrid(k1,k2)
    # Polar coordiantes (complex phase space)
    r = np.absolute(k1+1j*k2)
    ind_sort = np.argsort(r.flatten())

    # Ordered 1 dimensinal arrays
    Si_sorted = S_image.flatten()[ind_sort]
    S_ring = S_image.flatten().copy()
    r_log10_sort = np.log10(r.flatten()[ind_sort])
    #bins
    
    r_min = np.floor(np.min(r_log10_sort[1:]))
    r_max = np.floor(np.max(r_log10_sort))
    r_range = np.r_[np.arange(r_min,r_max),r_max]
    r_l = []
    for i in range(len(r_range)):
        for j in range(bins):
            r_l.append(10**(r_range[i]+j/bins))  
    r_l = np.r_[-np.inf,np.log10(np.array(r_l))]   
    mat = np.array([r_log10_sort-rb>=0  for rb in r_l[1:]])
    # bin number array
    r_n_bin = np.sum(mat,axis=0)
#    print(np.unique(r_n_bin),r_n_bin[0],r_n_bin[1],r_n_bin[-1],r_log10_sort[1],r_log10_sort[-1])
#    print(r_n_bin.shape,r_log10_sort.shape,np.unique(r_n_bin).shape,r_l)
    # Find all pixels that fall within each radial bin.
    delta_bin = r_n_bin[1:] - r_n_bin[:-1]
    bin_ind = np.r_[0,np.where(delta_bin)[0]+1,len(r_n_bin)-1] # location of changes in bin
#    print(bin_ind)
    nr = bin_ind[1:] - bin_ind[:-1]# number of elements per bin 
#    print(nr)    
    # Cumulative sum to 2D spectra to find sum per bin
    csSim = np.cumsum(Si_sorted)
    tbin = csSim[bin_ind[1:]] - csSim[bin_ind[:-1]]
    S_ave = tbin/nr   
#    print(S_ave)    
    for i,rb in enumerate(np.unique(r_n_bin)):
        Si_sorted[r_n_bin==rb] = S_ave[i]
    S_ring[ind_sort] = Si_sorted   
    S_r = Spectra_r()
    S_r.S = np.reshape(S_ring,S_image.shape)#np.reshape(S_ring,S_image.shape)    
                
#    for i in range(len(r_bin10)-1):
#        ind0 = (r_log10_sort>r_bin10[i]) & (r_log10_sort<=r_bin10[i+1])
#        S_ring[ind0] = np.sum(Si_sorted[ind0])/np.sum(ind0)   

    # Initialization of the data   
    
#    S_r.S = np.reshape(S_ring,S_image.shape)

    if stat==True:
        S_stat = np.array([[np.std(Si_sorted[r_n_bin==r]), 
                            np.median(Si_sorted[r_n_bin==r]),
                            np.max(Si_sorted[r_n_bin==r]),
                            np.min(Si_sorted[r_n_bin==r])]
                            for r in np.unique(r_n_bin)])        
        S_r.std = S_stat[:,0]
        S_r.median = S_stat[:,1]
        S_r.max = S_stat[:,2]
        S_r.min = S_stat[:,3]
    
    return S_r
# In[]
    
def avetriangles(xy,z,tri):
    """ integrate scattered data, given a triangulation
    zsum, areasum = sumtriangles( xy, z, triangles )
    In:
        xy: npt, dim data points in 2d, 3d ...
        z: npt data values at the points, scalars or vectors
        triangles: ntri, dim+1 indices of triangles or simplexes, as from
                   http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    Out:
        zsum: sum over all triangles of (area * z at midpoint).
            Thus z at a point where 5 triangles meet
            enters the sum 5 times, each weighted by that triangle's area / 3.
        areasum: the area or volume of the convex hull of the data points.
            For points over the unit square, zsum outside the hull is 0,
            so zsum / areasum would compensate for that.
            Or, make sure that the corners of the square or cube are in xy.
    """
    # z concave or convex => under or overestimates
    ind = ~np.isnan(z)
    xy = xy[ind,:]
    z = z[ind]
    triangles = Triangulation(xy[:,0],xy[:,1]).triangles
    npt, dim = xy.shape
    ntri, dim1 = triangles.shape
    assert npt == len(z), "shape mismatch: xy %s z %s" % (xy.shape, z.shape)
    assert dim1 == dim+1, "triangles ? %s" % triangles.shape
    zsum = np.zeros( z[0].shape )
    areasum = 0
    dimfac = np.prod( np.arange( 1, dim+1 ))
    for tri in triangles:
        corners = xy[tri]
        t = corners[1:] - corners[0]
        if dim == 2:
            area = abs( t[0,0] * t[1,1] - t[0,1] * t[1,0] ) / 2
        else:
            area = abs( np.linalg.det( t )) / dimfac  # v slow
        aux = area * np.nanmean(z[tri],axis=0)
        if ~np.isnan(aux):
            zsum += aux
            areasum += area
    return zsum/areasum
     
# In[]   
def upsample2 (x, k):
  """
  Upsample the signal to the new points using a sinc kernel. The
  interpolation is done using a matrix multiplication.
  Requires a lot of memory, but is fast.
  input:
  xt    time points x is defined on
  x     input signal column vector or matrix, with a signal in each row
  xp    points to evaluate the new signal on
  output:
  y     the interpolated signal at points xp
  """
  mn = x.shape
  if len(mn) == 2:
    m = mn[0]
    n = mn[1]
  elif len(mn) == 1:
    m = 1
    n = mn[0]
  else:
    raise ValueError ("x is greater than 2D")
  nn = n * k
  [T, Ts]  = np.mgrid[1:n:nn*1j, 1:n:n*1j]
  TT = Ts - T
  del T, Ts
  y = np.sinc(TT).dot (x.reshape(n, 1))
  return y.squeeze()   

# In[Plots]
def plot_log2D(k_int_grd, S, label_S = "$\log_{10}{S}$", C = 10**-4,fig_num='a'):
    
    k_log_1 = np.sign(k_int_grd[0])*np.log10(1+np.abs(k_int_grd[0])/C)#np.log10(np.abs(k_int_grd[0]))   
    k_log_2 = np.sign(k_int_grd[1])*np.log10(1+np.abs(k_int_grd[1])/C)#np.log10(np.abs(k_int_grd[1]))
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.use_sticky_edges = False
    S_lim = np.max(np.log10(S))
    print(S_lim)
    im=ax.contourf(k_log_1,k_log_2,np.log10(S),np.linspace(-3,S_lim,10),cmap='jet')
    ax.set_xlabel('$k_1$', fontsize=18)
    ax.set_ylabel('$k_2$', fontsize=18)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=18)
    
    fig.canvas.draw()
    
    xticks = np.max(ax.get_xlim())
    xticks = np.sign(xticks)*C*(10**(np.abs(xticks))-1)
    xticks = np.sign(xticks)*np.log10(np.abs(xticks))
    xticks = np.arange(np.ceil(np.log10(C)),np.ceil(xticks))
    xticks1 = 10**xticks    
    xticks = np.r_[xticks[::-1],-np.inf,xticks]
    xticks1 = np.r_[-xticks1[::-1],0,xticks1]
    xticks1 = np.sign(xticks1)*np.log10(1+np.abs(xticks1)/C);
    
    yticks = np.max(ax.get_ylim())
    yticks = np.sign(yticks)*C*(10**(np.abs(yticks))-1)
    yticks = np.sign(yticks)*np.log10(np.abs(yticks))
    yticks = np.arange(np.ceil(np.log10(C)),np.ceil(yticks))
    yticks1 = 10**yticks
    yticks = np.r_[yticks[::-1],-np.inf,yticks]
    yticks1 = np.r_[-yticks1[::-1],0,yticks1]
    yticks1 = np.sign(yticks1)*np.log10(1+np.abs(yticks1)/C);
       
    xticklabels = ['$-10^{'+str(int(xt))+'}$' if xt1 < 0 else '$10^{'+str(int(xt))+'}$' if xt1 > 0 else '0' for xt,xt1 in zip(xticks,xticks1)]
    yticklabels = ['$-10^{'+str(int(xt))+'}$' if xt1 < 0 else '$10^{'+str(int(xt))+'}$' if xt1 > 0 else '0' for xt,xt1 in zip(yticks,yticks1)]
    
    ax.set_xticks(xticks1)
    #print(xticks1)
    ax.set_xlim(-2,2)
    ax.set_yticks(yticks1)
    ax.set_ylim(-2,2)
    
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    
    ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel(label_S, fontsize=18)
    ax.text(0.05, 0.95, fig_num, transform=ax.transAxes, fontsize=18,verticalalignment='top')
    return []  
# In[Lanczos polar]  
def lanczos_kernel(r,r_1 = 1.22,r_2 = 2.233,a=1):
    kernel = lambda r: 2*sp.special.jv(1,a*np.pi*r)/r/np.pi/a
    kernel_w = kernel(r)*kernel(r*r_1/r_2)
    kernel_w[np.abs(r)>=r_2] = 0.0
    return kernel_w
    
def lanczos_int_sq(grid,tree,U,a=1):    
    dx = np.max(np.diff(grid[0].flatten()))
    dy = np.max(np.diff(grid[1].flatten()))
    X = grid[0].flatten()
    Y = grid[1].flatten()
    tree_grid = KDTree(np.c_[X,Y])
    d, n  = tree.query(tree_grid.data, k=40, return_distance = True)
    d=d/np.sqrt(dx*dy)
    S = np.sum(lanczos_kernel(d)*U[n],axis=1)
    S = np.reshape(S,grid[0].shape)
    return S
