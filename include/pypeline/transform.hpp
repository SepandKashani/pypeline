// ############################################################################
// transform.hpp
// =============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

/*
 * Coordinate transforms.
 */

#ifndef PYPELINE_TRANSFORM_HPP
#define PYPELINE_TRANSFORM_HPP

#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "eigen3/Eigen/Eigen"

#include "pypeline/types.hpp"

namespace pypeline { namespace transform {
    /**
     * Polar coordinates to Cartesian coordinates.
     *
     * Parameters
     * ----------
     * colat : Eigen::ArrayBase<Derived_colat> const&
     *     (N_height, 1) Polar/Zenith angle [rad].
     * lon : Eigen::ArrayBase<Derived_lon> const&
     *     (1, N_width) Longitude angle [rad].
     *
     * Returns
     * -------
     * XYZ : pypeline::ArrayXX_t<Derived_colat::Scalar>
     *     (3, N_height * N_width) Cartesian XYZ coordinates.
     *     This shape is obtained by reshaping the original (3, N_height, N_width) array.
     *
     */
    template <typename Derived_colat, typename Derived_lon>
    pypeline::ArrayXX_t<typename Derived_colat::Scalar> pol2cart(Eigen::ArrayBase<Derived_colat> const& colat,
                                                                 Eigen::ArrayBase<Derived_lon> const& lon) {
        static_assert(std::is_same<typename Derived_colat::Scalar,
                                   typename Derived_lon::Scalar>::value,
                      "Type[Derived_colat::Scalar, Derived_lon::Scalar] must be identical.");

        size_t const N_colat = colat.size();
        if (! ((colat.rows() == N_colat) && (colat.cols() == 1))) {
            std::string msg = "Parameter[colat] must be (N_height, 1).";
            throw std::runtime_error(msg);
        }

        size_t const N_lon = lon.size();
        if (! ((lon.rows() == 1) && (lon.cols() == N_lon))) {
            std::string msg = "Parameter[lon] must be (1, N_width).";
            throw std::runtime_error(msg);
        }

        using T_t = typename Derived_colat::Scalar;
        pypeline::ArrayXX_t<T_t> _sin_colat = colat.sin();
        pypeline::ArrayXX_t<T_t> _cos_colat = colat.cos();
        pypeline::ArrayXX_t<T_t> _sin_lon = lon.sin();
        pypeline::ArrayXX_t<T_t> _cos_lon = lon.cos();

        // Broadcast to correct shapes
        using Stride_t = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
        using Map_t = Eigen::Map<pypeline::ArrayXX_t<T_t>, Eigen::Unaligned, Stride_t>;
        Map_t sin_colat(_sin_colat.data(), N_colat, N_lon, Stride_t(1, 0));
        Map_t cos_colat(_cos_colat.data(), N_colat, N_lon, Stride_t(1, 0));
        Map_t sin_lon(_sin_lon.data(), N_colat, N_lon, Stride_t(0, 1));
        Map_t cos_lon(_cos_lon.data(), N_colat, N_lon, Stride_t(0, 1));

        pypeline::ArrayXX_t<T_t> X = sin_colat * cos_lon;
        pypeline::ArrayXX_t<T_t> Y = sin_colat * sin_lon;
        pypeline::ArrayXX_t<T_t> Z = cos_colat;

        pypeline::ArrayXX_t<T_t> XYZ(3, N_colat * N_lon);
        XYZ.row(0) = Eigen::Map<pypeline::ArrayX_t<T_t>>(X.data(), N_colat * N_lon);
        XYZ.row(1) = Eigen::Map<pypeline::ArrayX_t<T_t>>(Y.data(), N_colat * N_lon);
        XYZ.row(2) = Eigen::Map<pypeline::ArrayX_t<T_t>>(Z.data(), N_colat * N_lon);
        return XYZ;
    }
}}
#endif //PYPELINE_TRANSFORM_HPP
