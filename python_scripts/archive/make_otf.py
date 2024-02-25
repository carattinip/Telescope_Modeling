# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 09:50:00 2022

@author: thecainfamily
"""
import numpy as np
from scipy.fft import fft2, ifft2, fftshift

def make_otf(r1,r2,si,scale,phase):
    print('made it to make_otf')
    pupil=np.zeros((si,si))
    for ii in range(0,si-1):
        for jj in range(0,si-1):
            if((np.ceil(si/2)-np.floor(si/2))>0):
               dist=np.sqrt(np.power(np.round(ii-si/2),2)+np.power(np.round(jj-si/2),2))
            if((np.ceil(si/2)-np.floor(si/2))==0):
               dist=np.sqrt(np.power(np.round(ii-si/2-1),2)+np.power(np.round(jj-si/2-1),2))
            if(dist>r2):
                if(dist<r1):
                    pupil[ii,jj]=1
    cpupil=np.multiply(pupil,np.exp(1j*phase))
    psf=fft2(cpupil)
    psf=abs(psf)
    psf=np.multiply(psf, psf)
    spsf=np.sum(psf)
    norm_psf=scale*psf/spsf
    otf=fft2(norm_psf)        
    return(otf,norm_psf,pupil)

