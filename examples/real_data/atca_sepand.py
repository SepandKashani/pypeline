# ###############################################################
# Atca.py
# ================
# Author : Dewan Arun Singh (19884240@student.westernsydney.edu.au)
# ###################################################################

import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
from tqdm import tqdm as ProgressBar

import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.AtcaData as data
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_fd
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.measurement_set as measurement_set

# Instrument
ms_file = "/home/das/ATNF_data/ms_data/2051-377-2868.ms/"
ms = data.AtcaData(ms_file)
gram = bb_gr.GramBlock()

# Observation
FoV = np.deg2rad(0.15)  # Sepand: TBD from Miriad image. Conservative lower-bound.
# luc = coord.SkyCoord('20h55m37.085s -37d33m41.75s')
# rbc = coord.SkyCoord('20h54m11.789s -37d33m57.90s')
# FoV = luc.separation(rbc).to_value(u.rad)
channel_id = 257
frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)
sky_model = source.from_tgss_catalog(ms.field_center, FoV, N_src=1)


# Imaging
N_level = 4
N_bits = 32  # Sepand: This was the main issue.
N = ms.instrument.nyquist_rate(wl)
px_grid = grid.uniform(
    direction=ms.field_center.cartesian.xyz.value, FoV=FoV, size=[256, 256]
)  # [1920, 1080]


### Intensity Field ===========================================================
# Parameter Estimation
I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=1)
for t, f, S in ProgressBar(
    ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, 100), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, _ = measurement_set.filter_data(S, W)

    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()

# Imaging
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_level, N_bits)
for t, f, S in ProgressBar(
    ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, 1), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V, c_idx = I_dp(S, G)
    _ = I_mfs(D, V, XYZ.data, W.data, c_idx)
I_std, I_lsq = I_mfs.as_image()

### Sensitivity Field =========================================================
# Parameter Estimation
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=1)
for t in ProgressBar(ms.time["TIME"][::200]):
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)

    S_est.collect(G)
N_eig = S_est.infer_parameters()

# Imaging
S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
S_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits)
for t, f, S in ProgressBar(
    ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, 50), column="DATA")
):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, W = measurement_set.filter_data(S, W)

    D, V = S_dp(G)
    _ = S_mfs(D, V.astype(complex), XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))

_, S = S_mfs.as_image()


# Plot Results ================================================================
fig, ax = plt.subplots(ncols=2)
I_std_eq = s2image.Image(I_std.data / S.data, I_std.grid)
I_std_eq.draw(catalog=sky_model.xyz.T, data_kwargs=dict(cmap="cubehelix"), ax=ax[0])
ax[0].set_title("Bluebild Standardized Image")

I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid)
I_lsq_eq.draw(catalog=sky_model.xyz.T, data_kwargs=dict(cmap="cubehelix"), ax=ax[1])
ax[1].set_title("Bluebild Least-Squares Image")
fig.show()
