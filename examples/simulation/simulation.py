#####################################################################
# Simulation.py
# ================
# Author : Dewan Arun Singh (19884240@student.westernsydney.edu.au)
#####################################################################

"""
Common file for simulation of all telescopes. Removes the requirement for interacting with scripts
directly. Passing the required arguments would be enough to simulate using different values and telescopes.
"""

def simulate(start, fc, FoV, freq, N_station, telescope,variant=None):
    """
    interative function for simulating telescopes in pypeline.
   -----------------------------------------
    Parameters
    ----------------------------------------
    start : float
            A starting point for observation time
    fc : float
        list of azimuthal and elevation in the format [azimuthal,elevation]
    FoV : float
        Field of view of the telescope.
    freq :  Frequency for the simulation in Hz
    N_station : int
                 Number of stations you are trying to image.
    telescope : string
                Name of the telescope. Currently supported telescopes are: atca, lofar, & mwa.
    variant : string
            Only applies if you are imaging ATCA telescope. Is string consisting of one of the possible 20 configurations in the ATCA configuration array.
    
    ----------------------------------------

    """
    TELESCOPES = ['atca','lofar','mwa']
    if not telescope.lower() in TELESCOPES:
        raise ValueError (f"The value in telescope argument should be one of {TELESCOPES}")
    
    def atca():
        from examples.simulation.atca import atca
        atca(start, fc, FoV, freq, N_station,variant)
    
    def lofar():
        raise NotImplementedError

    def mwa():
        raise NotImplementedError

    switcher = {
        'atca' : atca,
        'lofar' : lofar,
        'mwa' : mwa, 
    }
   

    func = switcher.get(telescope,"Nothing") 
    func()        

