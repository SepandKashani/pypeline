.. ############################################################################
.. programming_guide.rst
.. =====================
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################


Programming Guide
=================

Pypeline is designed to be used as a Python3 package, but some sub-modules are written in C++
(``libpypeline.so``) for performance reasons.
The `Pybind11 <https://pybind11.readthedocs.io/en/stable/>`_ library is used to make all required
C++ functionality available from Python3.

As a user of Pypeline, it is important to take note of the information below.

Passing NumPy arrays to C++ code
--------------------------------

C++ functions taking NumPy arrays as parameters use the buffer protocol to read the array's content
without making any copies.
These tensor proxys can then be transparently used on the C++-side to perform numerical operations.

* C++ functions that *do not* modify the input arrays have no limitations whatsoever.
* C++ functions that *do* modify the input arrays only work on contiguous NumPy arrays.
  It is the responsibility of the user to guarantee this when calling a C++ function.
  Pypeline makes no checks on this front for you.
