    # #################################################################
   # AtcaTelescope.py                                               #
  # ================                                              #
 # Author : Dewan Arun Singh (dewanarunsingh@outlook.com)       #
# #############################################################

import numpy as np
import pandas as pd
import ImoT_tools.imot_tools.util.argcheck as chk
from pypeline.phased_array.instrument import EarthBoundInstrumentGeometryBlock

class AtcaTelescope (EarthBoundInstrumentGeometryBlock):
    """
    Australian Telescope Compact Array (ATCA) <https://www.narrabri.atnf.csiro.au/> - Located in Narrabi, NSW, Australia

    ATCA consists of a single station, with 6 antennas out of which 5 are movable and can be positioned at 44 different locations
    along a 3KM East-West track and a 214m North-South track.
    """
    @chk.check(dict(chk.)

    def __init__(self, array_config = "6A", ):
        """
        Parameters
        _______________
        array_config : string
        One of the possible 20 configurations of the antennas at ATCA.
        ______________________________________________________________
        """
        
    