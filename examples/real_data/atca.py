# ###############################################################
# Atca.py
# ================
# Author : Dewan Arun Singh (19884240@student.westernsydney.edu.au)
# ###################################################################

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
FoV = np.deg2rad(0.5)
channel_id = 257
frequency = ms.channels["FREQUENCY"][channel_id]
wl = constants.speed_of_light / frequency.to_value(u.Hz)
# sky_model = source.from_tgss_catalog(ms.field_center, FoV, N_src=1)


# Imaging
N_level = 4
N_bits = 32
N = ms.instrument.nyquist_rate(wl)
px_grid = grid.uniform(direction=ms.field_center.cartesian.xyz.value, FoV=FoV, size=[1920, 1080])


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

#############################   Periodic Synthesis  ###########################################
# # Imaging
# N_level = 4
# N_bits = 32
# R = ms.instrument.icrs2bfsf_rot(obs_start, obs_end)
# _, _, pix_colat, pix_lon = grid.equal_angle(
#     N=ms.instrument.nyquist_rate(wl),
#     direction=R @ ms.field_center.cartesian.xyz.value,  # BFSF-equivalent f_dir.
#     FoV=FoV,
# )
# N_FS, T_kernel = ms.instrument.bfsf_kernel_bandwidth(wl, obs_start, obs_end), np.deg2rad(10)

# ### Intensity Field ===========================================================
# # Parameter Estimation
# I_est = bb_pe.IntensityFieldParameterEstimator(N_level, sigma=0.95)
# for t, f, S in ProgressBar(
#     ms.visibilities(
#         channel_id=[channel_id], time_id=slice(None, None, 200), column="DATA"
#     )
# ):
#     wl = constants.speed_of_light / f.to_value(u.Hz)
#     XYZ = ms.instrument(t)
#     W = ms.beamformer(XYZ, wl)
#     G = gram(XYZ, W, wl)
#     S, _ = measurement_set.filter_data(S, W)

#     I_est.collect(S, G)
# N_eig, c_centroid = I_est.infer_parameters()

# # Imaging
# I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid)
# I_mfs = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, N_level, N_bits)
# for t, f, S in ProgressBar(
#     ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, 1), column="DATA_SIMULATED")
# ):
#     wl = constants.speed_of_light / f.to_value(u.Hz)
#     XYZ = ms.instrument(t)
#     W = ms.beamformer(XYZ, wl)
#     G = gram(XYZ, W, wl)
#     S, W = measurement_set.filter_data(S, W)

#     D, V, c_idx = I_dp(S, G)
#     _ = I_mfs(D, V, XYZ.data, W.data, c_idx)
# I_std, I_lsq = I_mfs.as_image()

# ### Sensitivity Field =========================================================
# # Parameter Estimation
# S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=0.95)
# for t in ProgressBar(ms.time["TIME"][::200]):
#     XYZ = ms.instrument(t)
#     W = ms.beamformer(XYZ, wl)
#     G = gram(XYZ, W, wl)

#     S_est.collect(G)
# N_eig = S_est.infer_parameters()

# # Imaging
# S_dp = bb_dp.SensitivityFieldDataProcessorBlock(N_eig)
# S_mfs = bb_fd.Fourier_IMFS_Block(wl, pix_colat, pix_lon, N_FS, T_kernel, R, 1, N_bits)
# for t, f, S in ProgressBar(
#     ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, 50), column="DATA_SIMULATED")
# ):
#     wl = constants.speed_of_light / f.to_value(u.Hz)
#     XYZ = ms.instrument(t)
#     W = ms.beamformer(XYZ, wl)
#     G = gram(XYZ, W, wl)
#     S, W = measurement_set.filter_data(S, W)

#     D, V = S_dp(G)
#     _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
# _, S = S_mfs.as_image()

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=2)
# I_std_eq = s2image.Image(I_std.data / S.data, I_std.grid)
# #I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax[0])
# ax[0].set_title("Bluebild Standardized Image")

I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid)
# I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax[1])
I_lsq_eq.draw(ax=ax[0])
ax[0].set_title("Bluebild Least-Squares Image")
fig.show()
