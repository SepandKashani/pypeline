// ############################################################################
// util.hpp
// ========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

/*
 * General helper functions.
 */

#ifndef PYPELINE_UTIL_HPP
#define PYPELINE_UTIL_HPP

#include <cmath>

namespace pypeline { namespace util {
    /**
     * Parameters
     * ----------
     * x : T const&
     *     Angle [deg]
     *
     * Returns
     * -------
     * y : T
     *     Angle [rad]
     */
    template <typename T>
    T deg2rad(T const& x) {
        T y = x * static_cast<T>(M_PI / 180.0);
        return y;
    }
}}
#endif //PYPELINE_UTIL_HPP
