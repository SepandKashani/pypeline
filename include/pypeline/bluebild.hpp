// ############################################################################
// bluebild.hpp
// ============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

/*
 * Bluebild field synthesizers that work in Fourier Series domain.
 */

#ifndef PYPELINE_BLUEBILD_HPP
#define PYPELINE_BLUEBILD_HPP

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "eigen3/Eigen/Eigen"
#include "eigen3/Eigen/Sparse"

#include "pypeline/ffs.hpp"
#include "pypeline/types.hpp"
#include "pypeline/util.hpp"

namespace pypeline { namespace bluebild {
    template <typename TT>
    class FourierFieldSynthesizerBlock {
        private:
            TT m_wl = 0;
            TT m_alpha_window = 0;
            TT m_T = 0;
            TT m_Tc = 0;
            TT m_mps = 0;

            size_t m_NFS = 0;
            size_t m_2N1Q = 0;
            size_t m_Nantenna = 0;
            size_t m_Neig = 0;
            size_t m_Nheight = 0;
            size_t m_Nwidth = 0;

            TT* m_grid_colat = nullptr;
            TT* m_grid_lon = nullptr;
            TT* m_R = nullptr;
            TT* m_XYZk = nullptr;
            TT* m_stat = nullptr;

            pypeline::ffs::FFS<TT>* m_K_transform = nullptr;
            pypeline::ffs::FFS<TT>* m_E_transform = nullptr;

            void print_internal_state() {
                std::cout << "m_wl = " << m_wl << std::endl;
                std::cout << "m_alpha_window = " << m_alpha_window << std::endl;
                std::cout << "m_T = " << m_T << std::endl;
                std::cout << "m_Tc = " << m_Tc << std::endl;
                std::cout << "m_mps = " << m_mps << std::endl;

                std::cout << "m_NFS = " << m_NFS << std::endl;
                std::cout << "m_2N1Q = " << m_2N1Q << std::endl;
                std::cout << "m_Nantenna = " << m_Nantenna << std::endl;
                std::cout << "m_Neig = " << m_Neig << std::endl;
                std::cout << "m_Nheight = " << m_Nheight << std::endl;
                std::cout << "m_Nwidth = " << m_Nwidth << std::endl;

                std::cout << "m_grid_colat = " << m_grid_colat << std::endl;
                std::cout << "m_grid_lon = " << m_grid_lon << std::endl;
                std::cout << "m_R = " << m_R << std::endl;
                std::cout << "m_XYZk = " << m_XYZk << std::endl;
                std::cout << "m_stat = " << m_stat << std::endl;

                std::cout << "m_K_transform = " << m_K_transform << std::endl;
                std::cout << "m_K_transform = " << m_E_transform << std::endl;
            }

            template <typename Derived_V, typename Derived_XYZ, typename Derived_W>
            bool _have_matching_shapes(Eigen::ArrayBase<Derived_V> const& V,
                                       Eigen::ArrayBase<Derived_XYZ> const& XYZ,
                                       Eigen::SparseMatrixBase<Derived_W> const& W) {
                if (V.rows() != W.cols()) {
                    return false;
                }
                if (W.rows() != XYZ.rows()) {
                    return false;
                }
                return true;
            }

            /**
             * Angular shift w.r.t kernel antenna coordinates.
             *
             * Parameters
             * ----------
             * XYZ : Eigen::ArrayBase<Derived_XYZ> const&
             *     (N_antenna, 3) Cartesian instrument geometry.
             *     `XYZ` must be given in BFSF.
             *
             * Returns
             * -------
             * theta : TT
             *     Angular shift (radians) such that ``dot(_XYZk, R(theta).T) == XYZ``.
             */
            template <typename Derived_XYZ>
            TT _phase_shift(Eigen::ArrayBase<Derived_XYZ> const& XYZ) {
                namespace _linalg = pypeline::linalg;

                Eigen::Map<pypeline::ArrayXX_t<TT>> _XYZk(m_XYZk, m_Nantenna, 3);
                pypeline::ArrayXX_t<TT> R_T = (_XYZk.matrix()
                                               .bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                                               .solve(XYZ.matrix()));

                pypeline::ArrayXX_t<TT> R(3, 3);
                R << R_T(0, 0), R_T(1, 0), 0,
                     R_T(0, 1), R_T(1, 1), 0,
                             0,         0, 1;
                TT theta = _linalg::z_rot2angle(R);
                return theta;
            }

            bool _regen_required(TT const& shift) {
                namespace _util = pypeline::util;
                TT const lhs = _util::deg2rad<TT>(-0.1);  // Slightly below 0 due to numerical rounding

                if ((lhs <= shift) && (shift <= m_mps)) {
                    return false;
                } else {
                    return true;
                }
            }

            template <typename Derived>
            void _regen_kernel(Eigen::ArrayBase<Derived> const& XYZ) {
                namespace _ffs = pypeline::ffs;
                namespace _transform = pypeline::transform;
                namespace _func = pypeline::func;

                pypeline::ArrayX_t<TT> lon_smpl = _ffs::ffs_sample<TT>(m_T, m_NFS, m_Tc, m_2N1Q);
                Eigen::Map<pypeline::ArrayXX_t<TT>> _grid_colat(m_grid_colat, m_Nheight, 1);
                pypeline::ArrayXX_t<TT> pix_smpl = _transform::pol2cart(_grid_colat, lon_smpl);

                pypeline::ArrayX_t<TT> XYZ_mean = XYZ.colwise().mean();
                pypeline::ArrayXX_t<TT> XYZ_c = (XYZ.rowwise() - XYZ_mean) * (2 * M_PI / m_wl);

                Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> pre_FSk(m_K_transform->data_in(),
                                                                          m_Nantenna,
                                                                          m_Nheight * m_2N1Q);
                pre_FSk.real() = (XYZ_c.matrix() * pix_smpl.matrix()).array().cos();
                pre_FSk.imag() = (XYZ_c.matrix() * pix_smpl.matrix()).array().sin();

                Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> window_FSk(m_K_transform->data_in(),
                                                                             m_Nantenna * m_Nheight,
                                                                             m_2N1Q);
                pypeline::ArrayXX_t<TT> window = _func::Tukey<TT>(m_T, m_Tc, m_alpha_window)(lon_smpl);
                // TODO: for some reason this does not work?
                // window_FSk.rowwise() *= window;

                m_K_transform->ffs();
                if (m_XYZk == nullptr) {
                    m_XYZk = new TT[3 * m_Nantenna];
                    Eigen::Map<pypeline::ArrayXX_t<TT>> _XYZk(m_XYZk, m_Nantenna, 3);
                    _XYZk = XYZ;
                }
            }

        public:
            /**
             * Parameters
             * ----------
             * wl : TT const&
             *     Wavelength [m] of observations.
             * grid_colat : pypeline::ArrayXX_t<TT> const&
             *     (N_height, 1) BFSF polar angles [rad].
             * grid_lon : pypeline::ArrayXX_t<TT> const&
             *     (1, N_width) equi-spaced BFSF azimuthal angles [rad].
             * N_FS : size_t const&
             *     :math:`2\pi`-periodic kernel bandwidth. (odd-valued)
             * T : TT const&
             *     Kernel periodicity [rad] to use for imaging.
             * R : pypeline::ArrayXX_t<TT> const&
             *     (3, 3) ICRS -> BFSF rotation matrix.
             * N_antenna : size_t const&
             *     Number of antennas.
             * N_eig : size_t const&
             *     Number of eigfunctions to output.
             * N_threads : size_t const&
             *     Number of threads to use.
             * effort : planning_effort
             *
             * Notes
             * -----
             * * `grid_colat` and `grid_lon` should be generated using :py:func:`~imot_tools.math.sphere.grid.equal_angle`.
             * * `N_FS` can be optimally chosen by calling :py:meth:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock.bfsf_kernel_bandwidth`.
             * * `R` can be obtained by calling :py:meth:`~pypeline.phased_array.instrument.EarthBoundInstrumentGeometryBlock.icrs2bfsf_rot`.
             */
            FourierFieldSynthesizerBlock(TT const& wl,
                                         pypeline::ArrayXX_t<TT> const& grid_colat,
                                         pypeline::ArrayXX_t<TT> const& grid_lon,
                                         size_t const& N_FS,
                                         TT const& T,
                                         pypeline::ArrayXX_t<TT> const& R,
                                         size_t const& N_antenna,
                                         size_t const& N_eig,
                                         size_t const& N_threads,
                                         pypeline::ffs::planning_effort effort) {
                { // Check argument validity ==================================
                    if (wl <= 0) {
                        std::string msg = "Parameter[wl] must be positive.";
                        throw std::runtime_error(msg);
                    }

                    size_t const N_colat = grid_colat.size();
                    if (! ((grid_colat.rows() == N_colat) && (grid_colat.cols() == 1))) {
                        std::string msg = "Parameter[grid_colat] must be (N_height, 1).";
                        throw std::runtime_error(msg);
                    }

                    size_t const N_lon = grid_lon.size();
                    if (! ((grid_lon.rows() == 1) && (grid_lon.cols() == N_lon))) {
                        std::string msg = "Parameter[grid_lon] must be (1, N_width).";
                        throw std::runtime_error(msg);
                    }

                    if (N_FS % 2 == 0) {
                        std::string msg = "Parameter[N_FS] must be odd-valued.";
                        throw std::runtime_error(msg);
                    }

                    if (! ((0 < T) && (T <= 2 * M_PI))) {
                        std::string msg = "Parameter[T] is out of bounds.";
                        throw std::runtime_error(msg);
                    }

                    if (! ((R.rows() == 3) && (R.cols() == 3))) {
                        std::string msg = "Parameter[R] must be (3, 3).";
                        throw std::runtime_error(msg);
                    }
                }

                { // Set all internal variables ===============================
                    m_wl = wl;
                    m_Nantenna = N_antenna;
                    m_Neig = N_eig;
                    m_Nheight = grid_colat.size();
                    m_Nwidth = grid_lon.size();

                    m_grid_colat = new TT[grid_colat.size()];
                    if (m_grid_colat == nullptr) {
                        std::string msg = "Failed to allocate Attribute[m_grid_colat].";
                        throw std::runtime_error(msg);
                    }
                    Eigen::Map<pypeline::ArrayXX_t<TT>> _grid_colat(m_grid_colat, grid_colat.size(), 1);
                    _grid_colat = grid_colat;

                    m_grid_lon = new TT[grid_lon.size()];
                    if (m_grid_lon == nullptr) {
                        std::string msg = "Failed to allocate Attribute[m_grid_lon].";
                        throw std::runtime_error(msg);
                    }
                    Eigen::Map<pypeline::ArrayXX_t<TT>> _grid_lon(m_grid_lon, 1, grid_lon.size());
                    _grid_lon = grid_lon;

                    m_R = new TT[R.size()];
                    if (m_R == nullptr) {
                        std::string msg = "Failed to allocate Attribute[m_R].";
                        throw std::runtime_error(msg);
                    }
                    Eigen::Map<pypeline::ArrayXX_t<TT>> _R(m_R, 3, 3);
                    _R = R;

                    if (std::abs(T - 2 * M_PI) < 1e-6) {
                        // No PeriodicSynthesis, but set params to still work.
                        m_alpha_window = 0;
                        m_T = 2 * M_PI;
                        m_Tc = M_PI;
                        m_mps = 2 * M_PI;
                        m_NFS = N_FS;
                    } else {
                        // PeriodicSynthesis
                        m_alpha_window = 0.1;
                        TT const T_min = (1 + m_alpha_window) * (grid_lon.maxCoeff() - grid_lon.minCoeff());
                        if (T < T_min) {
                            std::string msg = "Parameter[T] must be greater than T_min.";
                            throw std::runtime_error(msg);
                        }
                        m_T = T;

                        TT const lon_start = grid_lon(0, 0);
                        TT const lon_end = grid_lon(0, grid_lon.cols() - 1);
                        TT const T_start = lon_end + T * (0.5 * m_alpha_window - 1.0);
                        TT const T_end = lon_end + T * (0.5 * m_alpha_window);
                        m_Tc = 0.5 * (T_start + T_end);
                        m_mps = lon_start - (T_start + 0.5 * T * m_alpha_window);

                        size_t N_FS_trunc = std::ceil(static_cast<TT>(N_FS) * T / (2 * M_PI));
                        if (N_FS_trunc % 2 == 0) {
                            N_FS_trunc += 1;
                        }
                        m_NFS = N_FS_trunc;
                    }

                    m_2N1Q = pypeline::ffs::next_fast_len(m_NFS);
                    m_stat = new TT[m_Neig * m_Nheight * m_2N1Q];
                    if (m_stat == nullptr) {
                        std::string msg = "Failed to allocate Attribute[m_stat].";
                        throw std::runtime_error(msg);
                    }
                    Eigen::Map<pypeline::ArrayX_t<TT>> _stat(m_stat, m_Neig * m_Nheight * m_2N1Q);
                    _stat = pypeline::ArrayX_t<TT>::Zero(m_Neig * m_Nheight * m_2N1Q);

                    m_K_transform = new pypeline::ffs::FFS<TT>(std::vector<size_t>{m_Nantenna * m_Nheight, m_2N1Q}, 1, m_T, m_Tc, m_NFS, true, N_threads, effort);
                    if (m_K_transform == nullptr) {
                        std::string msg = "Failed to allocate Attribute[m_K_transform].";
                        throw std::runtime_error(msg);
                    }

                    m_E_transform = new pypeline::ffs::FFS<TT>(std::vector<size_t>{m_Neig * m_Nheight, m_2N1Q}, 1, m_T, m_Tc, m_NFS, true, N_threads, effort);
                    if (m_E_transform == nullptr) {
                        std::string msg = "Failed to allocate Attribute[m_E_transform].";
                        throw std::runtime_error(msg);
                    }
                }

                print_internal_state();
            }

            ~FourierFieldSynthesizerBlock() {
                if (m_grid_colat != nullptr) {
                    delete m_grid_colat;
                    m_grid_colat = nullptr;
                }

                if (m_grid_lon != nullptr) {
                    delete m_grid_lon;
                    m_grid_lon = nullptr;
                }

                if (m_R != nullptr) {
                    delete m_R;
                    m_R = nullptr;
                }

                if (m_XYZk != nullptr) {
                    delete m_XYZk;
                    m_XYZk = nullptr;
                }

                if (m_stat != nullptr) {
                    delete m_stat;
                    m_stat = nullptr;
                }

                if (m_K_transform != nullptr) {
                    delete m_K_transform;
                    m_K_transform = nullptr;
                }

                if (m_E_transform != nullptr) {
                    delete m_E_transform;
                    m_E_transform = nullptr;
                }
            }

            /**
             * Compute instantaneous field statistics.
             *
             * Parameters
             * ----------
             * V : Eigen::ArrayBase<Derived_V> const& V
             *     (N_beam, N_eig) complex-valued eigenvectors.
             * XYZ : Eigen::ArrayBase<Derived_XYZ> const& V
             *     (N_antenna, 3) Cartesian instrument geometry.
             *     `XYZ` must be given in ICRS.
             * W : Eigen::SparseMatrixBase<Derived_W> const&
             *     (N_antenna, N_beam) synthesis beamweights.
             *
             * Notes
             * -----
             * Results available in internal variable `m_stat`.
             */
            template <typename Derived_V, typename Derived_XYZ, typename Derived_W>
            void operator()(Eigen::ArrayBase<Derived_V> const& V,
                            Eigen::ArrayBase<Derived_XYZ> const& XYZ,
                            Eigen::SparseMatrixBase<Derived_W> const& W) {
                if(! _have_matching_shapes(V, XYZ, W)) {
                    std::string msg = "Parameters[V, XYZ, W] are inconsistent.";
                    throw std::runtime_error(msg);
                }

                Eigen::Map<pypeline::ArrayXX_t<TT>> _R(m_R, 3, 3);
                pypeline::ArrayXX_t<TT> bfsf_XYZ = XYZ.matrix() * _R.matrix().transpose();
                TT phase_shift = 0;
                if (m_XYZk == nullptr) {
                    phase_shift = 1e8;  // Just need something unreasonably large
                } else {
                    phase_shift = _phase_shift(bfsf_XYZ);
                }
                std::cout << "theta = " << phase_shift << std::endl;

                if (_regen_required(phase_shift)) {
                    _regen_kernel(bfsf_XYZ);
                    phase_shift = 0.0;
                }
            }
    };
}}
#endif //PYPELINE_BLUEBILD_HPP
