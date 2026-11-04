import glob

from astropy.io import fits 
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from photutils.aperture import CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_2dg

from pyklip.klip import meas_contrast 

# given file structure is folder labeled "pwfc_on", "pwfc_off", and corresponding darks 
# also we know that images are 264x256 
# 5 files in each folder 

path_to = "not_committing_my_filpath_to_github" 
dark_files = glob.glob(path_to + 'darks/*.fits')
pwfc_on_files = glob.glob(path_to + 'pwfc_on/*.fits')
pwfc_off_files = glob.glob(path_to + 'pwfc_off/*.fits')

dark_array = np.zeros((5, 264, 256), dtype=float)
pwfc_on_array = np.zeros((5, 264, 256), dtype=float)
pwfc_off_array = np.zeros((5, 264, 256), dtype=float)

for index, dark in enumerate(dark_files):
    dark_array[index, :, :] = fits.getdata(dark)

for index, pwfc_on_frame in enumerate(pwfc_on_files):
    pwfc_on_array[index, :, :] = fits.getdata(pwfc_on_frame)

for index, pwfc_off_frame in enumerate(pwfc_off_files):
    pwfc_off_array[index, :, :] = fits.getdata(pwfc_off_frame)

superdark = np.median(dark_array, axis=0)
pwfc_on_stack = np.median(pwfc_on_array, axis=0)
pwfc_off_stack = np.median(pwfc_off_array, axis=0)

pwfc_image = pwfc_on_stack - superdark 
integrator_image = pwfc_off_stack - superdark 

# lam/D at 2190nm / 10 meter telescope / 9.94 mas/pix scale 
# is 4.5 pixels 
seps, contrast_int = meas_contrast(integrator_image, iwa=0, owa=256/2, resolution=4.5)
_, contrast_pwfc = meas_contrast(pwfc_image, iwa=0, owa=256/2, resolution=4.5)

plt.semilogy(seps*9.94, contrast_int, color='grey', label='integrator')
plt.semilogy(seps*9.94, contrast_pwfc, color='cyan', label='predictor')
plt.xlabel('separation [mas]')
plt.ylabel('contrast')
plt.savefig('integrator_vs_predictor_corongagraph_free_contrast.png')
plt.clf()


print('center of pwfc data at ', centroid_2dg(pwfc_image))
print('center of integrator data at ', centroid_2dg(integrator_image))
# center of the pwfc_off data is 119, 132
# center of the pwfc_on data is 119, 131 
# brightest pix will give 119, 131 for both but subpixel does 131 --> 131.9 

apers_pwfc = []
for r_in in np.arange(1, 119, 1):
    aper = CircularAnnulus([119, 131], r_in, r_in+1)
    apers_pwfc.append(aper)

apers_int = []
for r_in in np.arange(1, 119, 1):
    aper = CircularAnnulus([119, 132], r_in, r_in+1)
    apers_int.append(aper)

phots_pwfc = []
for aper in apers_pwfc:
    phot = aperture_photometry(pwfc_image, aper)
    phots_pwfc.append(float(phot['aperture_sum']/aper.area))

phots_int = []
for aper in apers_int:
    phot = aperture_photometry(integrator_image, aper)
    phots_int.append(float(phot['aperture_sum']/aper.area))

radial_seps = [aper.r_in*9.94 for aper in apers_pwfc]

plt.semilogy(radial_seps, phots_int, color='grey', label='integrator')
plt.semilogy(radial_seps, phots_pwfc, color='cyan', label='predicotr')
plt.xlabel('separation [mas]')
plt.ylabel('radial average [photons/pixel]')
plt.savefig('integrator_vs_predictor_radial_average.png')
plt.clf()

plt.imshow(pwfc_image[131-60:131+60, 119-60:119+60], norm=colors.LogNorm(vmin=2e2, vmax=50000), cmap='cividis')
plt.colorbar()
plt.axis('off')
plt.savefig('coronagraph_free_contrast_predictor.png')
plt.clf()

plt.imshow(integrator_image[132-60:132+60, 119-60:119+60], norm=colors.LogNorm(vmin=2e2, vmax=50000), cmap='cividis')
plt.colorbar()
plt.axis('off')
plt.savefig('coronagraph_free_contrast_integrator.png')
plt.clf()
