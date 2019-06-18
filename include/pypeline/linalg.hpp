// ############################################################################
// linalg.hpp
// ==========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

/*
 * Linear algebra routines.
 */

#ifndef PYPELINE_LINALG_HPP
#define PYPELINE_LINALG_HPP

#include <cmath>
#include <stdexcept>
#include <string>

#include "eigen3/Eigen/Eigen"

#include "pypeline/util.hpp"

namespace pypeline { namespace linalg {
    /**
     * Determine rotation angle from Z-axis rotation matrix.
     *
     * Parameters
     * ----------
     * R : Eigen::ArrayBase<Derived> const&
     *     (3, 3) rotation matrix around the Z-axis.
     *
     * Returns
     * -------
     * angle : Derived::Scalar
     *     Signed rotation angle [rad].
     */
    template <typename Derived>
    typename Derived::Scalar z_rot2angle(Eigen::ArrayBase<Derived> const& R) {
        namespace _util = pypeline::util;

        if (!((R.rows() == 3) && (R.cols() == 3))) {
            std::string const msg = "Parameter[R] must have shape (3, 3).";
            throw std::runtime_error(msg);
        }
        // if not np.allclose(R[[0, 1, 2, 2, 2], [2, 2, 2, 0, 1]], np.r_[0, 0, 1, 0, 0]):
        //     raise ValueError("Parameter[R] is not a rotation matrix around the Z-axis.")

        using T = typename Derived::Scalar;
        T const ct = _util::clip<T>(R(0, 0), -1.0, 1.0);
        T const st = _util::clip<T>(R(1, 0), -1.0, 1.0);
        T angle = static_cast<T>(0.0);
        if (st >= static_cast<T>(0.0)) {  // In quadrants I or II
            angle = std::acos(ct);
        } else {  // In quadrants III or IV
            angle = - std::acos(ct);
        }

        return angle;
    }
}}
#endif //PYPELINE_LINALG_HPP
