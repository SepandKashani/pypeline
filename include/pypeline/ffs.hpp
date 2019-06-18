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

#include <complex>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "eigen3/Eigen/Eigen"
#include "fftw3.h"
#include "pybind11/pybind11.h"
#include "pybind11/embed.h"

#include "pypeline/types.hpp"
#include "pypeline/util.hpp"

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

    enum class planning_effort: unsigned int {
        NONE = FFTW_ESTIMATE,
        MEASURE = FFTW_MEASURE
    };

    /*
     * FFTW wrapper to plan 1D complex->complex (i)FFTs on multi-dimensional tensors.
     */
    template <typename T>
    class FFT {
        private:
            static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                          "T only accepts {float, double}.");
            static constexpr bool is_float = std::is_same<T, float>::value;
            using fftw_data_t = std::conditional_t<is_float, fftwf_complex, fftw_complex>;
            using fftw_plan_t = std::conditional_t<is_float, fftwf_plan, fftw_plan>;

            fftw_plan_t m_plan_fft;
            fftw_plan_t m_plan_fft_r;
            fftw_plan_t m_plan_ifft;
            fftw_plan_t m_plan_ifft_r;
            std::complex<T> *m_data_in = nullptr;
            std::complex<T> *m_data_out = nullptr;
            size_t m_axis = 0;
            std::vector<size_t> m_shape {};

            void setup_threads(size_t const N_threads) {
                if (is_float) {
                    fftwf_init_threads();
                    fftwf_plan_with_nthreads(N_threads);
                } else {
                    fftw_init_threads();
                    fftw_plan_with_nthreads(N_threads);
                }
            }

            void allocate_buffers(bool const inplace) {
                size_t N_cells = 1;
                for (size_t len_dim : m_shape) {N_cells *= len_dim;}

                m_data_in = reinterpret_cast<std::complex<T>*>(fftw_malloc(sizeof(std::complex<T>) * N_cells));
                if (inplace) {
                    m_data_out = m_data_in;
                } else {
                    m_data_out = reinterpret_cast<std::complex<T>*>(fftw_malloc(sizeof(std::complex<T>) * N_cells));
                }

                if (m_data_in == nullptr) {
                    std::string msg = "Could not allocate input buffer.";
                    throw std::runtime_error(msg);
                } else {
                    Eigen::Map<pypeline::ArrayX_t<std::complex<T>>> data_in(m_data_in, N_cells);
                    data_in = pypeline::ArrayX_t<std::complex<T>>::Zero(N_cells);
                }

                if (m_data_out == nullptr) {
                    std::string msg = "Could not allocate output buffer.";
                    throw std::runtime_error(msg);
                } else {
                    Eigen::Map<pypeline::ArrayX_t<std::complex<T>>> data_out(m_data_out, N_cells);
                    data_out = pypeline::ArrayX_t<std::complex<T>>::Zero(N_cells);
                }
            }

            void allocate_plans(planning_effort const effort) {
                // Determine right planning function to use based on T.
                using fftw_plan_func_t = fftw_plan_t (*)(int, const fftw_iodim *,
                                                         int, const fftw_iodim *,
                                                         fftw_data_t *, fftw_data_t *,
                                                         int, unsigned int);
                fftw_plan_func_t plan_func;
                if (is_float) {
                    plan_func = (fftw_plan_func_t) &fftwf_plan_guru_dft;
                } else {
                    plan_func = (fftw_plan_func_t) &fftw_plan_guru_dft;
                }

                // Fill in Guru interface's parameters. =======================
                std::vector<size_t> shape(m_shape);
                std::vector<size_t> strides(shape.size(), 1);  // unit = elements, not bytes
                std::partial_sum(shape.rbegin(),
                                 shape.rend() - 1,
                                 strides.rbegin() + 1,
                                 std::multiplies<int>());

                int const rank = 1;
                fftw_iodim dims_info {static_cast<int>(shape[m_axis]),
                                      static_cast<int>(strides[m_axis]),
                                      static_cast<int>(strides[m_axis])};
                std::vector<fftw_iodim> dims{dims_info};

                int const howmany_rank = shape.size() - 1;
                std::vector<fftw_iodim> howmany_dims(howmany_rank);
                for (size_t i = 0, j = 0; i < static_cast<size_t>(howmany_rank); ++i, ++j) {
                    if (i == m_axis) {j += 1;}

                    fftw_iodim info {static_cast<int>(shape[j]),
                                     static_cast<int>(strides[j]),
                                     static_cast<int>(strides[j])};
                    howmany_dims[i] = info;
                }

                fftw_data_t *data_in = reinterpret_cast<fftw_data_t*>(m_data_in);
                fftw_data_t *data_out = reinterpret_cast<fftw_data_t*>(m_data_out);
                // ============================================================

                m_plan_fft = plan_func(rank, dims.data(),
                                       howmany_rank, howmany_dims.data(),
                                       data_in, data_out, FFTW_FORWARD,
                                       static_cast<unsigned int>(effort));
                m_plan_fft_r = plan_func(rank, dims.data(),
                                         howmany_rank, howmany_dims.data(),
                                         data_out, data_in, FFTW_FORWARD,
                                         static_cast<unsigned int>(effort));
                m_plan_ifft = plan_func(rank, dims.data(),
                                        howmany_rank, howmany_dims.data(),
                                        data_in, data_out, FFTW_BACKWARD,
                                        static_cast<unsigned int>(effort));
                m_plan_ifft_r = plan_func(rank, dims.data(),
                                          howmany_rank, howmany_dims.data(),
                                          data_out, data_in, FFTW_BACKWARD,
                                          static_cast<unsigned int>(effort));

                if (m_plan_fft == nullptr) {
                    std::string msg = "Could not plan fft() transform.";
                    throw std::runtime_error(msg);
                }
                if (m_plan_fft_r == nullptr) {
                    std::string msg = "Could not plan fft_r() transform.";
                    throw std::runtime_error(msg);
                }
                if (m_plan_ifft == nullptr) {
                    std::string msg = "Could not plan ifft() transform.";
                    throw std::runtime_error(msg);
                }
                if (m_plan_ifft_r == nullptr) {
                    std::string msg = "Could not plan ifft_r() transform.";
                    throw std::runtime_error(msg);
                }
            }

        public:
            /*
             * Parameters
             * ----------
             * shape : std::vector<size_t> const&
             *     Dimensions of input/output arrays.
             * axis : size_t const&
             *     Dimension along which to apply transform.
             * inplace : bool const&
             *     Perform in-place transforms.
             *     If enabled, only one array will be internally allocated.
             * N_threads : size_t const&
             *     Number of threads to use.
             * effort : planning_effort
             *
             * Notes
             * -----
             * Input and output buffers are initialized to 0 by default.
             */
            FFT(std::vector<size_t> const& shape,
                size_t const& axis,
                bool const& inplace,
                size_t const& N_threads,
                planning_effort effort):
                m_axis(axis), m_shape(shape) {
                if (shape.size() < 1) {
                    std::string msg = "Parameter[shape] cannot be empty.";
                    throw std::runtime_error(msg);
                }

                if (axis >= shape.size()) {
                    std::string msg = "Parameter[axis] must be lie in {0, ..., shape.size()-1}.";
                    throw std::runtime_error(msg);
                }

                if (N_threads < 1) {
                    std::string msg = "Parameter[N_threads] must be positive.";
                    throw std::runtime_error(msg);
                }

                setup_threads(N_threads);
                allocate_buffers(inplace);
                allocate_plans(effort);
            }

            ~FFT() {
                // Determine right destroy function to use based on T.
                using fftw_destroy_plan_func_t = void (*)(fftw_plan_t);
                fftw_destroy_plan_func_t destroy_plan_func;
                if (is_float) {
                    destroy_plan_func = (fftw_destroy_plan_func_t) &fftwf_destroy_plan;
                } else {
                    destroy_plan_func = (fftw_destroy_plan_func_t) &fftw_destroy_plan;
                }

                destroy_plan_func(m_plan_fft);
                destroy_plan_func(m_plan_fft_r);
                destroy_plan_func(m_plan_ifft);
                destroy_plan_func(m_plan_ifft_r);

                // Determine right free function to use based on T.
                using fftw_free_func_t = void (*)(fftw_data_t*);
                fftw_free_func_t free_func;
                if (is_float) {
                    free_func = (fftw_free_func_t) &fftwf_free;
                } else {
                    free_func = (fftw_free_func_t) &fftw_free;
                }

                bool out_of_place = (m_data_in != m_data_out);
                free_func(reinterpret_cast<fftw_data_t*>(m_data_in));
                if (out_of_place) {
                    free_func(reinterpret_cast<fftw_data_t*>(m_data_out));
                }
            }

            /*
             * Returns
             * -------
             * data_in : std::complex<T>*
             *     Pointer to input array.
             */
            std::complex<T>* data_in() {
                return m_data_in;
            }

            /*
             * Returns
             * -------
             * data_out : std::complex<T>*
             *     Pointer to output array.
             *     If `inplace` was set to true, then ``data_in() == data_out()``.
             */
            std::complex<T>* data_out() {
                return m_data_out;
            }

            /*
             * Returns
             * -------
             * shape : std::vector<size_t>
             *     Dimensions of the input buffers.
             */
            std::vector<size_t> shape() {
                return m_shape;
            }

            /*
             * Transform input buffer using 1D-FFT, result available in output buffer.
             */
            void fft() {
                // Determine right execute function to use based on T.
                using fftw_execute_func_t = void (*)(const fftw_plan_t);
                fftw_execute_func_t execute_func;
                if (is_float) {
                    execute_func = (fftw_execute_func_t) &fftwf_execute;
                } else {
                    execute_func = (fftw_execute_func_t) &fftw_execute;
                }

                execute_func(m_plan_fft);
            }

            /*
             * Transform output buffer using 1D-FFT, result available in input buffer.
             */
            void fft_r() {
                // Determine right execute function to use based on T.
                using fftw_execute_func_t = void (*)(const fftw_plan_t);
                fftw_execute_func_t execute_func;
                if (is_float) {
                    execute_func = (fftw_execute_func_t) &fftwf_execute;
                } else {
                    execute_func = (fftw_execute_func_t) &fftw_execute;
                }

                execute_func(m_plan_fft_r);
            }

            /*
             * Transform input buffer using 1D-iFFT, result available in output buffer.
             */
            void ifft() {
                // Determine right execute function to use based on T.
                using fftw_execute_func_t = void (*)(const fftw_plan_t);
                fftw_execute_func_t execute_func;
                if (is_float) {
                    execute_func = (fftw_execute_func_t) &fftwf_execute;
                } else {
                    execute_func = (fftw_execute_func_t) &fftw_execute;
                }

                execute_func(m_plan_ifft);

                // Correct FFTW's lack of scaling during iFFTs.
                size_t N_cells = 1;
                for (size_t len_dim : m_shape) {N_cells *= len_dim;}
                Eigen::Map<pypeline::ArrayX_t<std::complex<T>>> data(m_data_out, N_cells);

                T const scale = T(1.0) / static_cast<T>(m_shape[m_axis]);
                data *= scale;
            }

            /*
             * Transform output buffer using 1D-iFFT, result available in input buffer.
             */
            void ifft_r() {
                // Determine right execute function to use based on T.
                using fftw_execute_func_t = void (*)(const fftw_plan_t);
                fftw_execute_func_t execute_func;
                if (is_float) {
                    execute_func = (fftw_execute_func_t) &fftwf_execute;
                } else {
                    execute_func = (fftw_execute_func_t) &fftw_execute;
                }

                execute_func(m_plan_ifft_r);

                // Correct FFTW's lack of scaling during iFFTs.
                size_t N_cells = 1;
                for (size_t len_dim : m_shape) {N_cells *= len_dim;}
                Eigen::Map<pypeline::ArrayX_t<std::complex<T>>> data(m_data_in, N_cells);

                T const scale = T(1.0) / static_cast<T>(m_shape[m_axis]);
                data *= scale;
            }

            std::string __repr__() {
                namespace _util = pypeline::util;

                std::stringstream msg;
                msg << "FFT<" << ((is_float) ? "float" : "double") << ">("
                    << "shape=" << _util::print(m_shape) << ", "
                    << "axis=" << std::to_string(m_axis) << ", "
                    << "inplace=" << ((m_data_in == m_data_out) ? "true" : "false")
                    << ")";

                return msg.str();
            }
    };

    /*
     * FFTW implementation of the Fast Fourier Series (FFS) algorithm.
     */
    template <typename TT>
    class FFS {
        private:
            static constexpr bool is_float = std::is_same<TT, float>::value;

            size_t m_axis = 0;
            std::vector<size_t> m_shape {};
            FFT<TT> m_transform;
            std::vector<std::complex<TT>> m_mod_1;
            std::vector<std::complex<TT>> m_mod_2;

            void compute_modulation_vectors(TT const& T,
                                            TT const& T_c,
                                            size_t const& N_FS) {
                int const N_samples = static_cast<int>(m_shape[m_axis]);
                int const M = N_samples / 2;
                int const N = static_cast<int>(N_FS) / 2;

                std::complex<TT> _1j(0, 1);
                pypeline::ArrayX_t<TT> E_1(N_samples);
                E_1 << pypeline::ArrayX_t<TT>::LinSpaced(N_FS, -N, N),
                       pypeline::ArrayX_t<TT>::Zero(N_samples - N_FS);
                std::complex<TT> B_2 = std::exp(-_1j * TT(2 * M_PI / N_samples));

                std::complex<TT> B_1;
                pypeline::ArrayX_t<TT> E_2 = pypeline::ArrayX_t<TT>::Zero(N_samples);
                if (N_samples % 2 == 1) {
                    B_1 = std::exp(_1j * TT(2 * M_PI / T) * TT(T_c));
                    E_2 << pypeline::ArrayX_t<TT>::LinSpaced(M + 1, 0, M),
                           pypeline::ArrayX_t<TT>::LinSpaced(M, -M, -1);
                } else {
                    B_1 = std::exp(_1j * TT(2 * M_PI / T) * TT(T_c + (T / (2 * N_samples))));
                    E_2 << pypeline::ArrayX_t<TT>::LinSpaced(M, 0, M - 1),
                           pypeline::ArrayX_t<TT>::LinSpaced(M, -M, -1);
                }

                for (size_t i = 0; i < N_samples; ++i) {
                    std::complex<TT> const _m1 = std::pow<TT>(B_1, - E_1(i));
                    m_mod_1.push_back(_m1);

                    std::complex<TT> const _m2 = std::pow<TT>(B_2, - TT(N) * E_2(i));
                    m_mod_2.push_back(_m2);
                }
            }

        public:
            /*
             * Parameters
             * ----------
             * shape : std::vector<size_t> const&
             *     Dimensions of input/output arrays.
             * axis : size_t const&
             *     Dimension along which function samples are stored.
             * T : TT const&
             *     Function period.
             * T_c : TT const&
             *     Period mid-point.
             * N_FS : int const&
             *     Function bandwidth.
             * inplace : bool const&
             *     Perform in-place transforms.
             *     If enabled, only one array will be internally allocated.
             * N_threads : size_t const&
             *     Number of threads to use.
             * effort : planning_effort
             *
             * Notes
             * -----
             * Input and output buffers are initialized to 0 by default.
             */
            FFS(std::vector<size_t> const& shape,
                size_t const& axis,
                TT const& T,
                TT const& T_c,
                size_t const& N_FS,
                bool const& inplace,
                size_t const& N_threads,
                planning_effort effort):
                m_axis(axis), m_shape(shape),
                m_transform(shape, axis, inplace, N_threads, effort) {
                if (T <= 0) {
                    std::string msg = "Parameter[T] must be positive.";
                    throw std::runtime_error(msg);
                }
                if (N_FS % 2 == 0) {
                    std::string msg = "Parameter[N_FS] must be odd-valued.";
                    throw std::runtime_error(msg);
                }
                size_t const N_samples = shape[axis];
                if (!((3 <= N_FS) && (N_FS <= N_samples))) {
                    std::string msg = "Parameter[N_FS] must lie in {3, ..., shape[axis]}.";
                    throw std::runtime_error(msg);
                }

                // Current limitation: restrict to `shape.size() == 2` and `axis=1` due to Eigen.
                // All that needs to be changed for full tensor support is how the modulation
                // vectors are applied to the data in ffs(), iffs(), ffs_r(), iffs_r()
                if (!((shape.size() == 2) && (axis == 1))) {
                    std::string msg = "CODE_LIMITATION: Parameter[shape] must have length 2; axis=1";
                    throw std::runtime_error(msg);
                }

                compute_modulation_vectors(T, T_c, N_FS);
            }

            /*
             * Returns
             * -------
             * data_in : std::complex<TT>*
             *     Pointer to input array.
             */
            std::complex<TT>* data_in() {
                return m_transform.data_in();
            }

            /*
             * Returns
             * -------
             * data_out : std::complex<TT>*
             *     Pointer to output array.
             *     If `inplace` was set to true, then ``data_in() == data_out()``.
             */
            std::complex<TT>* data_out() {
                return m_transform.data_out();
            }

            /*
             * Returns
             * -------
             * shape : std::vector<size_t>
             *     Dimensions of the input buffers.
             */
            std::vector<size_t> shape() {
                return m_shape;
            }

            /*
             * Transform input buffer using 1D-FFS, result available in output buffer.
             *
             * It is assumed the input buffer contains function samples in the same order specified
             * by :cpp:func:`ffs_sample` along dimension `axis`.
             *
             * The output buffer will contain coefficients
             * :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_samples}`.
             */
            void ffs() {
                Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> data_in(this->data_in(), shape()[0], shape()[1]);
                Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> data_out(this->data_out(), shape()[0], shape()[1]);
                Eigen::Map<pypeline::ArrayX_t<std::complex<TT>>> mod_1(m_mod_1.data(), m_mod_1.size());
                Eigen::Map<pypeline::ArrayX_t<std::complex<TT>>> mod_2(m_mod_2.data(), m_mod_2.size());

                data_in.rowwise() *= mod_2;
                m_transform.fft();

                bool const out_of_place = (this->data_in() != this->data_out());
                if (out_of_place) {
                    // Undo in-place modulation by `mod_2`.
                    data_in.rowwise() *= mod_2.conjugate();
                }

                TT const N_samples = static_cast<int>(m_shape[m_axis]);
                data_out.rowwise() *= (mod_1 / N_samples);
            }

            /*
             * Transform output buffer using 1D-FFS, result available in input buffer.
             *
             * It is assumed the output buffer contains function samples in the same order specified
             * by :cpp:func:`ffs_sample` along dimension `axis`.
             *
             * The input buffer will contain coefficients
             * :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_samples}`.
             */
            void ffs_r() {
                Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> data_in(this->data_in(), shape()[0], shape()[1]);
                Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> data_out(this->data_out(), shape()[0], shape()[1]);
                Eigen::Map<pypeline::ArrayX_t<std::complex<TT>>> mod_1(m_mod_1.data(), m_mod_1.size());
                Eigen::Map<pypeline::ArrayX_t<std::complex<TT>>> mod_2(m_mod_2.data(), m_mod_2.size());

                data_out.rowwise() *= mod_2;
                m_transform.fft_r();

                bool const out_of_place = (this->data_in() != this->data_out());
                if (out_of_place) {
                    // Undo in-place modulation by `mod_2`.
                    data_out.rowwise() *= mod_2.conjugate();
                }

                TT const N_samples = static_cast<int>(m_shape[m_axis]);
                data_in.rowwise() *= (mod_1 / N_samples);
            }

            /*
             * Transform input buffer using 1D-iFFS, result available in output buffer.
             *
             * It is assumed the input buffer contains FS coefficients ordered as
             * :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_{samples}}`.
             * along dimension `axis`.
             *
             * The output buffer will contain the original function samples in the same order
             * specified by :cpp:func:`ffs_sample`.
             */
            void iffs() {
                Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> data_in(this->data_in(), shape()[0], shape()[1]);
                Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> data_out(this->data_out(), shape()[0], shape()[1]);
                Eigen::Map<pypeline::ArrayX_t<std::complex<TT>>> mod_1(m_mod_1.data(), m_mod_1.size());
                Eigen::Map<pypeline::ArrayX_t<std::complex<TT>>> mod_2(m_mod_2.data(), m_mod_2.size());

                data_in.rowwise() *= mod_1.conjugate();
                m_transform.ifft();

                bool const out_of_place = (this->data_in() != this->data_out());
                if (out_of_place) {
                    // Undo in-place modulataion by `mod_1`.
                    data_in.rowwise() *= mod_1;
                }

                TT const N_samples = static_cast<int>(m_shape[m_axis]);
                data_out.rowwise() *= mod_2.conjugate() * N_samples;
            }

            /*
             * Transform output buffer using 1D-iFFS, result available in input buffer.
             *
             * It is assumed the output buffer contains FS coefficients ordered as
             * :math:`\left[ x_{-N}^{FS}, \ldots, x_{N}^{FS}, 0, \ldots, 0 \right] \in \mathbb{C}^{N_{samples}}`.
             * along dimension `axis`.
             *
             * The input buffer will contain the original function samples in the same order
             * specified by :cpp:func:`ffs_sample`.
             */
            void iffs_r() {
                Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> data_in(this->data_in(), shape()[0], shape()[1]);
                Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> data_out(this->data_out(), shape()[0], shape()[1]);
                Eigen::Map<pypeline::ArrayX_t<std::complex<TT>>> mod_1(m_mod_1.data(), m_mod_1.size());
                Eigen::Map<pypeline::ArrayX_t<std::complex<TT>>> mod_2(m_mod_2.data(), m_mod_2.size());

                data_out.rowwise() *= mod_1.conjugate();
                m_transform.ifft_r();

                bool const out_of_place = (this->data_in() != this->data_out());
                if (out_of_place) {
                    // Undo in-place modulataion by `mod_1`.
                    data_out.rowwise() *= mod_1;
                }

                TT const N_samples = static_cast<int>(m_shape[m_axis]);
                data_in.rowwise() *= mod_2.conjugate() * N_samples;
            }

            std::string __repr__() {
                namespace _util = pypeline::util;

                std::stringstream msg;
                msg << "FFS<" << ((is_float) ? "float" : "double") << ">("
                    << "shape=" << _util::print(m_shape) << ", "
                    << "axis=" << std::to_string(m_axis) << ", "
                    << "inplace=" << ((data_out() == data_in()) ? "true" : "false")
                    << ")";

                return msg.str();
            }
    };
}}
#endif //PYPELINE_FFS_HPP
