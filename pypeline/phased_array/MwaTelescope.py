# ########################################################
# MwaTelescope.py                                        #
# ______________________                                 #
# Author : Dewan Arun Singh (dewanarunsingh@outlook.com) #
# ######################################################## 

import pathlib

import astropy.coordinates as coord
import astropy.time as time
import imot_tools.math.linalg as pylinalg
import imot_tools.math.special as sp
import imot_tools.math.sphere.transform as transform
import imot_tools.util.argcheck as chk
import numpy as np
import pandas as pd
import pkg_resources as pkg
import plotly.graph_objs as go
import scipy.linalg as linalg

import pypeline.core as core
import pypeline.util.array as array

from pypeline.phased_array.instrument import EarthBoundInstrumentGeometryBlock
from pypeline.phased_array.instrument import _as_InstrumentGeometry



class MwaBlock(EarthBoundInstrumentGeometryBlock):
    """
    `Murchison Widefield Array (MWA) <http://www.mwatelescope.org/>`_ located in Australia.

    MWA consists of 128 stations, each containing 16 dipole antennas.
    """

    @chk.check(dict(N_station=chk.allow_None(chk.is_integer), station_only=chk.is_boolean))
    def __init__(self, N_station=None, station_only=False):
        """
        Parameters
        ----------
        N_station : int
            Number of stations to use. (Default = all)

            Sometimes only a subset of an instrument’s stations are desired.
            Setting `N_station` limits the number of stations to those that appear first in `XYZ`
            when sorted by STATION_ID.

        station_only : bool
            If :py:obj:`True`, model MWA stations as single-element antennas. (Default = False)
        """
        XYZ = self._get_geometry(station_only)
        super().__init__(XYZ, N_station)

    def _get_geometry(self, station_only):
        """
        Load instrument geometry.

        Parameters
        ----------
        station_only : bool
            If :py:obj:`True`, model stations as single-element antennas.

        Returns
        -------
        :py:class:`~pypeline.phased_array.instrument.InstrumentGeometry`
            ITRS instrument geometry.
        """
        rel_path = pathlib.Path("data", "phased_array", "instrument", "MWA.csv")
        abs_path = pkg.resource_filename("pypeline", str(rel_path))

        itrs_geom = pd.read_csv(abs_path).set_index("STATION_ID")

        station_id = itrs_geom.index.get_level_values("STATION_ID")
        if station_only:
            itrs_geom.index = pd.MultiIndex.from_product(
                [station_id, [0]], names=["STATION_ID", "ANTENNA_ID"]
            )
        else:
            # Generate flat 4x4 antenna grid pointing towards the Noth pole.
            x_lim = y_lim = 1.65
            lY, lX = np.meshgrid(
                np.linspace(-y_lim, y_lim, 4), np.linspace(-x_lim, x_lim, 4), indexing="ij"
            )
            l = np.stack((lX, lY, np.zeros((4, 4))), axis=0)

            # For each station: rotate 4x4 array to lie on the sphere's surface.
            xyz_station = itrs_geom.loc[:, ["X", "Y", "Z"]].values
            df_stations = []
            for st_id, st_cog in zip(station_id, xyz_station):
                _, st_colat, st_lon = transform.cart2pol(*st_cog)
                st_cog_unit = transform.pol2cart(1, st_colat, st_lon).reshape(-1)

                R_1 = pylinalg.rot([0, 0, 1], st_lon)
                R_2 = pylinalg.rot(axis=np.cross([0, 0, 1], st_cog_unit), angle=st_colat)
                R = R_2 @ R_1

                st_layout = np.reshape(
                    st_cog.reshape(3, 1, 1) + np.tensordot(R, l, axes=1), (3, -1)
                )
                idx = pd.MultiIndex.from_product(
                    [[st_id], range(16)], names=["STATION_ID", "ANTENNA_ID"]
                )
                df_stations += [pd.DataFrame(data=st_layout.T, index=idx, columns=["X", "Y", "Z"])]
            itrs_geom = pd.concat(df_stations)

        XYZ = _as_InstrumentGeometry(itrs_geom)
        return XYZ
