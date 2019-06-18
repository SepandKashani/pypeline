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

#include <stdexcept>
#include <string>

#include "eigen3/Eigen/Eigen"
#include "pybind11/pybind11.h"
#include "pybind11/embed.h"

#include "pypeline/types.hpp"

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

    /**
     * Signal sample positions for :cpp:class:`pypeline::ffs::FFS`.
     *
     * Return the coordinates at which a signal must be sampled to use
     * :cpp:class:`pypeline::ffs::FFS`.
     *
     * Parameters
     * ----------
     * T : TT const&
     *     Function period.
     * N_FS : size_t const&
     *     Function bandwidth.
     * T_c : TT const&
     *     Period mid-point.
     * N_s : size_t const&
     *     Number of samples.
     *
     * Returns
     * -------
     * sample_point : pypeline::ArrayX_t<TT>
     *     (N_s,) coordinates at which to sample a signal (in the right order).
     */
    template <typename TT>
    pypeline::ArrayX_t<TT> ffs_sample(TT const& T,
                                      size_t const& N_FS,
                                      TT const& T_c,
                                      size_t const& N_s) {
        if (T <= static_cast<TT>(0.0)) {
            std::string const msg = "Parameter[T] must be postive.";
            throw std::runtime_error(msg);
        }
        if (N_FS < 3) {
            std::string const msg = "Parameter[N_FS] must be \ge 3.";
            throw std::runtime_error(msg);
        }
        if (N_s < N_FS) {
            std::string const msg = "Parameter[N_s] must be \ge Parameter[N_FS] (signal bandwidth).";
            throw std::runtime_error(msg);
        }

        pypeline::ArrayX_t<TT> sample_point(N_s);
        pypeline::ArrayX_t<TT> idx(N_s);
        if (N_s % 2 == 1) {  // Odd-valued
            int const M = (N_s - 1) / 2;
            idx << pypeline::ArrayX_t<TT>::LinSpaced(M + 1, 0, M),
                   pypeline::ArrayX_t<TT>::LinSpaced(M, -M, -1);
            sample_point = T_c + (T / N_s) * idx;
        } else {
            int const M = N_s / 2;
            idx << pypeline::ArrayX_t<TT>::LinSpaced(M, 0, M - 1),
                   pypeline::ArrayX_t<TT>::LinSpaced(M, -M, -1);
            sample_point = T_c + (T / N_s) * (static_cast<TT>(0.5) + idx);
        }

        return sample_point;
    }
}}
#endif //PYPELINE_FFS_HPP
