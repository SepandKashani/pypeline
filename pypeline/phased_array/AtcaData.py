    # #################################################################
   # AtcaData.py                                                    #
  # ================                                              #
 # Author : Dewan Arun Singh (dewanarunsingh@outlook.com)       #
# #############################################################

import numpy as np
import pandas as pd
import ImoT_tools.imot_tools.util.argcheck as chk
import casacore.tables as ct
import pypeline.phased_array.instrument as instrument

from pypeline.phased_array.measurement_set import MeasurementSet

class AtcaData(MeasurementSet):
    """ 
    Australia Telescope Compact Array (ATCA) data handler class.
    """
    @chk.check(dict(fileName = chk.is_instance(str)))

    def __init__(self, fileName):
        """
        Parameters
        ----------
        fileName : String
        Path to the MS file for the data.

        arrayConfig : int
        A number corresponding to one of the 28 array configurations at ATCA.
        """
        super().__init__(fileName)
        
        
    @property
    def instrument(self):
        """
        Returns
        -------
        :py:class:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock`
            Computes Instrument position.
        """
        if self._instrument is None:
            # Following the  MS file specification from, https://casa.nrao.edu/casadocs/casa-5.1.0/reference-material/measurement-set
            # Some remarks on the required fields:
            # - NAME: ID of the antenna
            # - POSITION: Absolute positions of the antennas
            # - FLAG_ROW: True/False value for each antenna. 
            #             If any of the polarization flags is True for a given antenna, then the
            #             antenna can be discarded from that configuration.

            query = f'select NAME,POSITION,FLAG_ROW from {self._msf}::ANTENNA'
            table = ct.taql(query)

            #Getting the necessary data out from the ms table
            antenna_name = table.getcol("NAME")
            antenna_position = table.getcol("POSITION")
            antenna_flag = table.getcol("FLAG_ROW")

            #Creating a dataframe that holds the all the relevant data and filtering out flagged antennas.
            antennae = np.reshape(antenna_position,(6,3))
            
            #Create a multi-level index for the new dataframe with STATION_ID always as 0
            cfg_index = pd.MultiIndex.from_product([{0},antenna_name],names=['STATION_ID','ANTENNA_NAME']
            )
            df = pd.DataFrame(data=antennae, columns=['X','Y','Z'], index=cfg_index).loc[~antenna_flag]

            #Creating instrument block for ATCA data specified.
            XYZ = instrument.InstrumentGeometry(xyz=df.values, ant_idx=df.index)
            self._instrument = instrument.EarthBoundInstrumentGeometryBlock(XYZ)
        
        return self._instrument
    
    



