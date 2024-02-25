"""
Author: 2nd Lt. Patrick Carattini
Created: 4 Feb 2023
Title: gbs_otf
"""

import numpy as np
from numpy import squeeze, ceil, real
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from skimage.util import random_noise
import os


from make_otf2 import make_otf, make_pupil
from simulate_moon import simulate_moon
from make_long_otf2 import make_long_otf
from detector_blur import detector_blur

#TODO: check for existince of otfs in the file system to automate generation of OTFs

# Telescope Parameters
D = 0.07                        # diameter of telescope in meters
obs = 0                         # obscuration diameter
lamb = 610*10**(-9)             # wavelength of ligth in meters
f = 0.4                         # focal length in meters
si = 3000                       # ccd pixel dimensions
pixle_per_meter = si/(2*D)          # array is 5x5 meter
dm = 1/pixle_per_meter          # distance between pixels in meters
r1 = (D/2)*pixle_per_meter      # radius of telescope in pixels
r2 = (obs/2)*pixle_per_meter    # radius of obscuration in pixels
scale = 1                       # value at dc
phase = np.zeros([si, si])      # zero for non-abberated system
dt = 100.0e-3                   # CCD integration time

# Atmosphere Parameters
z = 100*10**3                   # Karman line ~ 100km
ro = 0.02                       # seeing parameter
r1a = D/2                       # radius used for turbulence calculations
dx = 4*r1a/si                   # x steps for turbulence calculations

# paths TODO: make OS agnostic
source_file_path = os.getcwd() + '/source_files/'
figures_path = os.getcwd() + '/figures/'



generate = True    # flag to generate data or load from memory
gen_objects = True  # flag to generate dim objects around the moon

# Generate data 
if generate:

    # Model Telescope
    tele_pupil = make_pupil(r1,r2,si)
    [tele_otf, tele_psf] = make_otf(scale,tele_pupil)
    np.save(source_file_path + 'tele_otf', tele_otf)
    np.save(source_file_path + 'tele_pupil', tele_pupil)

    # Model Atmosphere
    atmosphere_otf = make_long_otf(r1a,dm,si,ro)
    np.save(source_file_path + 'atmosphere_otf', atmosphere_otf)

    # Model Telescope + Atmosphere
    turbulent_otf = np.multiply(tele_otf, atmosphere_otf)
    turbulent_psf = ifft2(fftshift(turbulent_otf))

    # Detector model ( adds blur so the niquist image is represented at 1/2 niquist)
    detector_otf = detector_blur(2, 2, si)
    np.save(source_file_path + 'detector_otf', detector_otf)

    # Telescope + Atmosphere + Detector Model
    total_otf = np.multiply(turbulent_otf,detector_otf)
    np.save(source_file_path + 'total_otf', total_otf)

    # Simulate Moon
    photon_img = simulate_moon(D, f, lamb, dt, gen_objects)
    np.save(source_file_path + 'photon_img', photon_img)

    # Model Moon as seen by the telescope at niquist
    output_img = real(ifft2(np.multiply(total_otf, fft2(photon_img))))
    np.save(source_file_path + 'output_img', output_img)

    # Down Sample to match IRL detector size 
    downscale_factor = 2
    if gen_objects:
        down_sample_img = output_img[:,::downscale_factor, ::downscale_factor]

    else:
        down_sample_img = output_img[::downscale_factor, ::downscale_factor]
    
    np.save(source_file_path + 'down_sample_img', down_sample_img)

    # Add Poison Noise
    noisy_img = np.random.poisson(output_img)
    np.save(source_file_path + 'noisy_img', noisy_img)

## Load From Files
else:

    tele_otf = np.load(source_file_path + 'tele_otf.npy')

    tele_pupil = np.load(source_file_path + 'tele_pupil.npy')

    atmosphere_otf = np.load(source_file_path + 'atmosphere_otf.npy')

    detector_otf = np.load(source_file_path + 'detector_otf.npy')

    total_otf = np.load(source_file_path + 'total_otf.npy')

    photon_img = np.load(source_file_path + 'photon_img.npy')

    output_img = np.load(source_file_path + 'output_img.npy')

    down_sample_img = np.load(source_file_path + 'down_sample_img.npy')

    noisy_img = np.load(source_file_path + 'noisy_img.npy')



# f, ax = plt.subplots(1,3)
# ax[0].imshow(output_img)
# ax[0].set_title('Img Out')
# ax[0].tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False) 
# ax[1].imshow(down_sample_img)
# ax[1].set_title('Down Sample Image')
# ax[1].tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False) 
# ax[2].imshow(noisy_img)
# ax[2].set_title('Noisy Img')
# ax[2].tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False) 
# plt.savefig(figures_path + 'detector sim plots')



# # Plots of Sim Chain
# f, axarr = plt.subplots(1,3)
# axarr[0].imshow(photon_img[1], norm='linear')
# axarr[0].set_xlabel('Simulated Moon')
# axarr[0].tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False) 
# axarr[1].imshow(output_img[1], norm='linear')
# axarr[1].set_xlabel('Telescope x Moon')
# axarr[1].tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False) 
# axarr[2].imshow(noisy_img[1], norm='linear')
# axarr[2].set_xlabel('Noisy Img')
# axarr[2].tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False) 
# plt.savefig('plots')

# Plots of Objects
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(noisy_img[0])
axarr[0,0].set_xlabel('left 1/4')
axarr[0,0].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

axarr[0,1].imshow(noisy_img[1])
axarr[0,1].set_xlabel('left 2/4')
axarr[0,1].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

axarr[1,0].imshow(noisy_img[2])
axarr[1,0].set_xlabel('right 2/4')
axarr[1,0].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
                
axarr[1,1].imshow(noisy_img[3])
axarr[1,1].set_xlabel('right 4/4')
axarr[1,1].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 


plt.savefig(figures_path + 'detector_model_imgs')