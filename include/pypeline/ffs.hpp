// ############################################################################
// ffs.hpp
// =======
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

/*
 * Fast Fourier Series tools.
 */

#ifndef PYPELINE_FFS_HPP
#define PYPELINE_FFS_HPP

#include "pybind11/pybind11.h"
#include "pybind11/embed.h"

namespace pypeline { namespace ffs {
    /**
     * Find the next fast size of input data to fft, for zero-padding, etc.
     *
     * SciPy’s FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this returns the next
     * composite of the prime factors 2, 3, and 5 which is greater than or equal to target. (These
     * are also known as 5-smooth numbers, regular numbers, or Hamming numbers.)
     *
     * Parameters
     * ----------
     * target : size_t const&
     *     Length to start searching from. Must be a positive size_teger.
     *
     * Returns
     * -------
     * out : size_t
     *     The first 5-smooth number greater than or equal to target.
     */
    size_t next_fast_len(size_t const& target) {
        pybind11::module fftpack = pybind11::module::import("scipy.fftpack");
        size_t out = fftpack.attr("next_fast_len")(target).cast<size_t>();

        return out;
    }
}}
#endif //PYPELINE_FFS_HPP
