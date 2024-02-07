# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 09:50:00 2022

@author: thecainfamily
"""
import numpy as np
from scipy.fft import fftshift
import matplotlib.pyplot as plt

def make_long_otf(r1,dx,si,ro):
    print('made it to make_long')
    otf=np.zeros((si,si))
    for ii in range(0,si-1):
        for jj in range(0,si-1):
            if((np.ceil(si/2)-np.floor(si/2))>0):
               dist=np.sqrt(np.power(np.round(ii-si/2),2)+np.power(np.round(jj-si/2),2))
            if((np.ceil(si/2)-np.floor(si/2))==0):
               dist=np.sqrt(np.power(np.round(ii-si/2-1),2)+np.power(np.round(jj-si/2-1),2))
            otf[ii,jj]=np.exp(-3.44*np.power((dx*dist/ro),5/3))
    otf2=fftshift(otf)
    return(otf2)

# long_otf=make_long_otf(5,10/400,400,2)


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# X = np.arange(0,len(long_otf))
# Y = np.arange(0,len(long_otf))
# X,Y = np.meshgrid(X, Y)

# ax.plot_surface(X,Y, fftshift(long_otf))
# plt.savefig('make_long')


