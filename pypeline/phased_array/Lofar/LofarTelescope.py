# #############################################################################
# LofarTelescope.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

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


class LofarBlock(EarthBoundInstrumentGeometryBlock):
    """
    `LOw-Frequency ARray (LOFAR) <http://www.lofar.org/>`_ located in Europe.

    This LOFAR model consists of 62 stations, each containing between 17 to 24 HBA dipole antennas.
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
            If :py:obj:`True`, model LOFAR stations as single-element antennas. (Default = False)
        """
        XYZ = self._get_geometry(station_only)
        super().__init__(XYZ, N_station)

    def _get_geometry(self, station_only):
        """
        Load instrument geometry.

        Returns
        -------
        :py:class:`~pypeline.phased_array.instrument.InstrumentGeometry`
            ITRS instrument geometry.
        """
        rel_path = pathlib.Path("data", "phased_array", "instrument", "LOFAR.csv")
        abs_path = pkg.resource_filename("pypeline", str(rel_path))

        itrs_geom = pd.read_csv(abs_path).set_index(["STATION_ID", "ANTENNA_ID"])

        if station_only:
            itrs_geom = itrs_geom.groupby("STATION_ID").mean()
            station_id = itrs_geom.index.get_level_values("STATION_ID")
            itrs_geom.index = pd.MultiIndex.from_product(
                [station_id, [0]], names=["STATION_ID", "ANTENNA_ID"]
            )

        XYZ = _as_InstrumentGeometry(itrs_geom)
        return XYZ
