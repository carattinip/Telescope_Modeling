import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from math import floor
from scipy.fft import fft2, ifft2, fftshift

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

source_file_path = os.getcwd() + '/source_files/'
figures_path = os.getcwd() + '/figures/'

## Plot Pupil
# tele_pupil = np.load(source_file_path + 'tele_pupil.npy')

# plt.imshow(tele_pupil, extent=[0,(si/pixle_per_meter), 0, (si/pixle_per_meter)])
# plt.xlabel('Meters')
# plt.ylabel('Meters')
# plt.title('Telescope Pupil')
# plt.savefig(figures_path + 'pupil_plot')

## Plot Tele PSF and OTF
tele_otf = np.load(source_file_path + 'tele_otf.npy')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
fmax = (D-obs)/(2*lamb*f)
fs = fmax*2
df = fs/si
X = np.arange(-fmax,fmax, df)/1e3
Y = X
X, Y = np.meshgrid(X, Y)


# Plot the surface.
ax.set_xlabel('x Spatial Frequency (kilocycles/m)')
ax.set_ylabel('y Spatial Frequency (kilocycles/m)')
ax.set_zlabel('Magnitude')
ax.set_title('Telescope OTF')
surf = ax.plot_surface(X, Y, abs(fftshift(tele_otf)), cmap=cm.winter)


plt.savefig(figures_path + 'tele_otf')

