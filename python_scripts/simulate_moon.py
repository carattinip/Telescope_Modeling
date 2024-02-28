"""
Author: 2nd Lt. Patrick Carattini
Created: 4 Feb 2023
Title: gbs_otf
"""

import numpy as np
import os
from scipy.fft import fft2, ifft2, fftshift
from numpy import squeeze, ceil, real
import matplotlib.pyplot as plt

def simulate_moon(D,F,lam,dt, gen_dim_obj):

    path = os.getcwd() + '/source_files/moon_img.txt'
    opo9914d = np.genfromtxt(path, delimiter=",").astype('float')
    img = np.zeros((1200, 1200, 3))
    img[:, :1200, 0] = opo9914d[:, :1200]
    img[:, :1200, 1] = opo9914d[:, 1200:2400]
    img[:, :1200, 2] = opo9914d[:, 2400:3600]

    dpix = lam * F / (2 * D)
    dtheta = lam / (2 * D)
    Moon_dist = 384400000.0                   # meters
    Moon_diameter = 3474800.0                 # meters
    Moon_pixels = Moon_diameter / (Moon_dist * dtheta)

    pixels = round(Moon_pixels)
    f_interp_moon = np.zeros([pixels, pixels]).astype('complex')
    fmoon = fftshift(fft2(squeeze(img[:,:, 2])))
    half_pix = ceil(pixels / 2).astype('int')
    f_interp_moon[half_pix - 600: half_pix + 600,
                    half_pix - 600: half_pix + 600] = fmoon
    moon = real(ifft2(fftshift(f_interp_moon)))
    Source_img = np.multiply(np.ones([3000, 3000]), moon[1, 1])
    Source_img[1501 - half_pix + 1: 1501 + half_pix,
                1501 - half_pix + 1: 1501 + half_pix] = moon

    # Calculate energy from the moon and
    Intensity = 1000.0                          # w / m ^ 2 power per unit area hitting the moon
    h = 6.62607015e-34                          # plancks constant
    c = 3.0e8                                   # speed of light in meters
    v = c / lam                                 # frequency of light
    moon_reflectivity = 0.10                    # moon's reflectivity is 10%
    photons_moon = (Intensity * ((dtheta*Moon_dist)**2) * dt * moon_reflectivity) / (h * v)
    # energy = (photons / (4.0 * np.pi * Moon_dist ** 2.0)) * np.pi * (D / 2.0)**2.0
    photons_telescope = (photons_moon*np.pi*(D/2)**2)/(2*np.pi*(Moon_dist)**2)

    # Make Image reflect real energy values

    # moon_max = np.max(np.max(Source_img))
    moon_max = Source_img.max()

    norm_moon = np.divide(Source_img, moon_max)

    photons_img = np.multiply(norm_moon, photons_telescope)

    if gen_dim_obj:
        # Add dim objects
        object_size = 10        # area of object in m^2
        obj_reflectivity = 1
        obj_photons = (Intensity * (object_size) * dt * obj_reflectivity) / (h * v)
        obj_photons_telescope = (obj_photons*np.pi*(D/2)**2)/(2*np.pi*(Moon_dist)**2)

        num_objects = 4

        images = np.zeros([num_objects,3000,3000])
        # obj_pos = np.arange(0,3000,2,dtype=int)
        obj_pos= [0,999,1999,2999]

        img_index = np.arange(0,num_objects,dtype=int)
        images[img_index] = photons_img
        images[img_index,1500,obj_pos] = obj_photons_telescope

    else:
        images = photons_img

    return images

# photon_img = simulate_moon(0.07, 0.4, 610.0*10**-9,1)

# plt.imshow(photon_img)
# plt.savefig('photon_moon')