// ############################################################################
// _test.cpp
// =========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"


float test(float x) {
    return x + 1;
}


void _test_bindings(pybind11::module &m) {
    m.def("test",
          &test,
          pybind11::arg("x").noconvert().none(false),
          pybind11::doc(R"EOF(
test(x)

Increment argument.

Parameters
----------
x : float
    Number to increment.

Returns
-------
y : float
    x + 1

Examples
--------
.. testsetup::

   from pypeline._pyFFS import test

.. doctest::

   >>> x = 1
   >>> y = test(x)

   >>> y == 2
   True
)EOF"));
}

PYBIND11_MODULE(_test, m) {
    pybind11::options options;
    options.disable_function_signatures();

    _test_bindings(m);
}
