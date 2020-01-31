    # #################################################################
   # AtcaTelescope.py                                               #
  # ================                                              #
 # Author : Dewan Arun Singh (dewanarunsingh@outlook.com)       #
# #############################################################


import numpy as np
import pandas as pd
import pkg_resources as pkg
import ImoT_tools.imot_tools.util.argcheck as chk

from pathlib import Path
from pypeline.phased_array.instrument import EarthBoundInstrumentGeometryBlock
from pypeline.phased_array.instrument import _as_InstrumentGeometry

class AtcaTelescope (EarthBoundInstrumentGeometryBlock):
    """
    Australian Telescope Compact Array (ATCA) <https://www.narrabri.atnf.csiro.au/> - Located in Narrabi, NSW, Australia

    ATCA consists of a single station, with 6 antennas out of which 5 are movable and can be positioned at 44 different locations
    along a 3KM East-West track and a 214m North-South track.
    """
    
    def __init__(self, array_config):
        """
        Parameters
        _______________
        array_config : int
        One of the possible 28 configurations of the antennas at ATCA. Each integer corresponds to an alphanumeric configuration
        as provided for ATCA.
        ______________________________________________________________
        """
        rel_path = Path("data", "phased_array", "instrument", "ATCA.csv")
        abs_path = pkg.resource_filename("pypeline", str(rel_path))

        if not Path(abs_path).is_file():
            raise FileNotFoundError
        else:
            # Get the ITRS geometry from the csv file and pass to super constructor. Not using "STATION_ONLY" as ATCA is a single
            # station telescope.
            itrs_geom = pd.read_csv(abs_path).set_index('VARIANT','ANTENNA_ID')
            XYZ = _as_InstrumentGeometry(itrs_geom)
        
        #Reconsidering using the super constructor as ATCA has only 1 station and might need to define it's own constructor. 
        # Thoughts??
        super().__init__(XYZ, array_config)

        
    