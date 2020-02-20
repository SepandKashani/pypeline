#####################################################################
# Atca.py
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
import imot_tools.util.argcheck as chk

import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.data_gen.statistics as statistics


import pypeline.phased_array.Atca.AtcaTelescope as instrument

@chk.check(dict(start=chk.is_instance(float),fc=chk.is_array_like,FoV=chk.is_instance(float),freq=chk.is_instance(float),
            N_station=chk.is_integer,telescope=chk.is_instance(str),variant=chk.is_instance(str)))
def atca(start, fc, FoV, freq, N_station, telescope,variant=None):
    # Observation
    obs_start = atime.Time(start, scale="utc", format="mjd")
    field_center=coord.SkyCoord(fc[0] * u.rad, fc[1] * u.rad)
    FoV = np.deg2rad(FoV)
    wl = constants.speed_of_light/freq
    # Instrument
    dev=instrument.AtcaTelescope(variant=variant)
    mb_cfg = [(_, _, field_center) for _ in range(N_station)]
    mb = beamforming.MatchedBeamformerBlock(mb_cfg)
    gram = bb_gr.GramBlock()

    # Data generation
    T_integration = 10
    sky_model = source.SkyEmission([(coord.SkyCoord('20h54m54.171s -37d33m51.11s', frame='icrs'),0.15)])
    vis = statistics.VisibilityGeneratorBlock(sky_model, T_integration, fs=196000, SNR=np.inf)
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
    for t in ProgressBar(time[::1]):
        XYZ = dev(t)
        W = mb(XYZ, wl)
        S = vis(XYZ, W, wl)
        G = gram(XYZ, W, wl)

        D, V, c_idx = I_dp(S, G)
        _ = I_mfs(D, V, XYZ.data, W.data, c_idx)
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
