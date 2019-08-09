#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:44:38 2019

@author: konstantinos

A script to check the use of KDTree

"""
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsRegressor

x=np.array([1,1,1,1,1,1])
y=np.array([1.5,2.5,4.5,7.5,14.5,30.5])
a=np.c_[x,y]
tree =KDTree(a)

x1=np.array([1,1,1,1,1,1])
y1=np.array([1,2,4,7,14,30])
a1=np.c_[x1,y1]
tree1 = KDTree(a1)

r=2
ind0,dist0 = tree.query_radius(tree1.data, r,return_distance = True,sort_results=True)# returns the indices of tree.data points that have distance<r from the tree1.data points
ind1,dist1 = tree1.query_radius(tree.data, r,return_distance = True,sort_results=True)# returns the indices of tree.data points that have distance<r from the tree1.data points

ind00=[]
ind01=[]
for i,j in zip(ind0,range(len(ind0))):
    if i.size > 0:
        ind00.append(np.asscalar(i[0]))
        ind01.append(j)     
    
ind11=[]
ind10=[]
for i,j in zip(ind1,range(len(ind1))):
    if i.size > 0:
        ind11.append(np.asscalar(i[0]))
        ind10.append(j)     
        
posg0=np.c_[0.5*(a[:,0][ind00]+a1[:,0][ind01]),0.5*(a[:,1][ind00]+a1[:,1][ind01])] 
posg1=np.c_[0.5*(a[:,0][ind10]+a1[:,0][ind11]),0.5*(a[:,1][ind10]+a1[:,1][ind11])]  
posg = np.vstack((posg0,posg1))
    
unique = [list(t) for t in zip(*list(set(zip(posg[:,0], posg[:,1]))))] 

posg = np.c_[unique[0],unique[1]]

treeg = KDTree(posg)

indg,distg = treeg.query_radius(treeg.data, r,return_distance = True,sort_results=True)
S = sorted(set((tuple(sorted(tuple(i))) for i in indg if len(tuple(i))>1)))#if the length of the tuple is more than 1, means that there is at least one neighbor other than the point itself,keeps the sets of the neighbors
nonS = [np.asscalar(i) for i in indg if len(tuple(i))==1] #they dont have any neighbors in distance <r
temp = [set(u) for u in S]
S = []

for ti in temp:
    aux = [t for t in temp if t!=ti]
    if not any(ti <= u for u in aux):
        print("not any(ti <= u for u in aux):")
        S.append(list(ti))
        
aux=np.array([np.mean(posg[list(p),:],axis=0) for p in S])  
posg = np.vstack((aux,posg[nonS])) 
tree_g = KDTree(posg)
    # Diastances and labels of neighbours to intersection points
d0,n0 = tree_g.query(tree.data, return_distance = True)
d1,n1 = tree_g.query(tree1.data, return_distance = True) 
dg,ng = tree_g.query(tree_g.data, k = 3, return_distance = True)

d0 = np.squeeze(d0)
d1 = np.squeeze(d1)
n0 = np.squeeze(n0)
n1 = np.squeeze(n1)
dg = np.squeeze(dg)
ng = np.squeeze(ng)

rg = dg[:,1]/2
n0_bigger = np.unique(n0[(np.max(np.c_[rg[n0],d0],axis=1)-d0 == 0).nonzero()[0]]) 
n1_bigger = np.unique(n0[(np.max(np.c_[rg[n0],d0],axis=1)-d0 == 0).nonzero()[0]]) 

for i in n0_bigger:
    rg[i] = np.max(d0[n0==i])
for i in n1_bigger:
    rg[i] = np.max([np.max(d1[n1==i]),rg[i]])        

rg=rg*1.01


#   Epanechnikov        

w0 = .75*(1-(d0/rg[n0])**2)
w1 = .75*(1-(d1/rg[n1])**2)

"""answer la4: "and this one? what for?""


#    w0 = (rg[n0]**2-d0**2)/(d0**2+rg[n0]**2)
#    w1 = (rg[n1]**2-d1**2)/(d1**2+rg[n1]**2)
# Delaunay triangulation of intersection points
#tri = Triangulation(posg[:,0], posg[:,1])  