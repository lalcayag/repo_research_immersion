#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:26:30 2019

@author: konstantinos
a script to check the  [item for sublist in raux for item in sublist]
"""
import numpy as np
r0=np.array([1,2,3,4,5])
raux=[]     
ind=[1,2,3]
raux.append(r0[ind])   
r_flat= [item for sublist in raux for item in sublist]
print(r_flat)