"""
Author: 2nd Lt. Patrick Carattini
Created: 4 Feb 2023
Title: gbs_otf
"""

import numpy as np
from numpy import squeeze, ceil, real
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
import os


from make_otf import make_otf
from make_long_otf import make_long_otf
from simulate_moon import simulate_moon

#TODO: check for existince of otfs in the file system to automate generation of OTFs

# Telescope Parameters
D = 0.07                        # diameter of telescope in meters
obs = 0                         # obscuration diameter
lamb = 610*10**(-9)             # wavelength of ligth in meters
f = 0.4                         # focal length in meters
si = 3000                       # ccd pixel dimensions
pixle_per_meter = si/5          # array is 5x5 meter
dm = 1/pixle_per_meter          # distance between pixels in meters
r1 = (D/2)*pixle_per_meter      # radius of telescope in pixels
r2 = (obs/2)*pixle_per_meter    # radius of obscuration in pixels
scale = 1                       # value at dc
phase = np.zeros([si, si])      # zero for non-abberated system

# path to saved files TODO: make OS agnostic
path = os.getcwd() + '/source_files/'

# Model Telescope
# tele_otf, norm_psf2 , pupil2 = make_otf(r1,r2,si,scale,phase)
# np.save('tele_otf', tele_otf)
tele_otf = np.load(path + 'tele_otf.npy')

# Atmosphere Parameters
z = 100*10**3                   # Karman line ~ 100km
ro = 0.02
r1 = D/2
dx = 4*r1/si


# Model Atmosphere
# atmosphere_otf = make_long_otf(r1,dm,si,ro)
# np.save('atmosphere_otf', atmosphere_otf)
atmosphere_otf = np.load(path + 'atmosphere_otf.npy')

# Model Telescope + Atmosphere
otfTurbulent = np.multiply(tele_otf, atmosphere_otf)
psfTurbulent = ifft2(fftshift(otfTurbulent))

# Simulate Moon
# photon_img = simulate_moon(0.07, 0.4, 610.0*10**-9,1)
# np.save('photon_img', photon_img)
photon_img = np.load(path + 'photon_img.npy')

output_img = real(ifft2(np.multiply(otfTurbulent, fft2(photon_img))))

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# X = np.arange(0,len(otfTurbulent))
# Y = np.arange(0,len(otfTurbulent))
# X,Y = np.meshgrid(X, Y)

# ax.plot_surface(X,Y, (abs(fftshift(fft2(photon_img)))))
# ax.plot_surface(X,Y, (abs(fftshift(atmosphere_otf))))
# plt.savefig('fft_photonimg')

# Add Poison Noise
noise_mask = np.random.poisson(real(output_img))
noisy_img = output_img + noise_mask

f, ax = plt.subplots(1,2)
ax[0].imshow(output_img)
ax[0].set_title('Img Out')
ax[1].imshow(noise_mask)
ax[1].set_title('Noise Mask')
plt.savefig('noisy plots')
# print(noise_mask.max())


# Plots
# f, axarr = plt.subplots(1,3)
# axarr[0].imshow(photon_img)
# axarr[0].set_title('Simulated Moon')
# axarr[1].imshow(output_img)
# axarr[1].set_title('Telescope x Moon')
# axarr[2].imshow(noisy_img)
# axarr[2].set_title('Noisy Img')
# plt.savefig('plots')