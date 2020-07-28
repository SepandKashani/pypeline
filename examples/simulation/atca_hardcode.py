#####################################################################
# atca_hardcode.py
# ================
# Author : Dewan Arun Singh (19884240@student.westernsydney.edu.au)
#####################################################################

"""
Simulated ATCA imaging with Bluebild (Standard Synthesis).
"""

from tqdm import tqdm as ProgressBar
import astropy.coordinates as coord
import astropy.time as atime
import astropy.units as u
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import imot_tools.math.sphere.transform as transform
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import pandas as pd

import imot_tools.util.argcheck as chk
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.data_gen.statistics as statistics


import pypeline.phased_array.Atca.AtcaTelescope as instrument



# Observation
obs_start = atime.Time(55679.80133020458, scale="utc", format="mjd")
field_center=coord.SkyCoord(-0.807622 * u.rad, -0.65556 * u.rad)
FoV,freq = np.deg2rad(0.39), 2867e+06
wl = constants.speed_of_light/freq
# Instrument
N_station=6
dev=instrument.AtcaTelescope(variant='6A')
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
#import pdb; pdb.set_trace()
gram = bb_gr.GramBlock()

# Data generation
T_integration = 10
sky_model = source.SkyEmission([(coord.SkyCoord('20h54m54.171s -37d33m51.11s', frame='icrs'),0.05)])
vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=np.inf)
#np.save('/home/das/Radio_Astronomy/synthetic_vis',vis)
time = obs_start + (T_integration * u.s) * np.arange(815)
#time = ms.time['TIME']
#import pdb; pdb.set_trace()
obs_end = time[-1]

# Imaging
N_level = 4
N_bits = 32

px_grid = grid.uniform(direction=field_center.cartesian.xyz.value,FoV=FoV,size=[600,600])


### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=1)
for t in ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    G = gram(XYZ, W, wl)
    #import pdb; pdb.set_trace()
    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()

# Imaging
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
I_mfs =bb_sd.Spatial_IMFS_Block(wl,px_grid,N_level,N_bits)
sim_data = np.empty([0,],dtype=np.complex64)
for t in ProgressBar(time[::1]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    S = vis(XYZ, W, wl)
    G = gram(XYZ, W, wl)
    if t==55679.8955431675:
        np.save('/home/das/Radio_Astronomy/simulated_visibility', S.data)
    sim = np.triu(S.data,k=1)
    i,j = [0,0,1,0,1,2,0,1,2,3,0,1,2,3,4],[1,2,2,3,3,3,4,4,4,4,5,5,5,5,5]
    for x,y in zip(i,j):
        sim_data = np.append(sim_data,sim[x,y])

    D, V, c_idx = I_dp(S, G)
    _ = I_mfs(D, V, XYZ.data, W.data, c_idx)
#np.save('/home/das/Radio_Astronomy/sim_data',sim_data)    
I_std, I_lsq = I_mfs.as_image()

### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=1)
for t in ProgressBar(time[::200]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
S_mfs = bb_sd.Spatial_IMFS_Block(wl,px_grid,1,N_bits)
for t in ProgressBar(time[::50]):
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)

    D, V = S_dp(G)
    _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
_, S = S_mfs.as_image()

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=2)
I_std_eq = s2image.Image(I_std.data / S.data, I_std.grid)
I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax[0],show_gridlines=False)
ax[0].set_title("Bluebild Standardized Image")

I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid)
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax[1], show_gridlines=False)
ax[1].set_title("Bluebild Least-Squares Image")
fig.show()