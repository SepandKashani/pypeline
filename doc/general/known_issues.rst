.. ############################################################################
.. known_issues.rst
.. ================
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################


Known Issues
============

* The test suite sometimes fails when launched in the same shell used to compile Pypeline's C++
  libraries.

* :py:class:`~pypeline.util.math.fourier.FFTW_xx` objects exposed through PyBind11 do not work when
  the input arrays have rank :math:`\gt` 2.  This limitation does not exist when using these
  functions directly from C++.
