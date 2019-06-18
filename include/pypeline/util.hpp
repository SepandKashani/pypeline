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
#include <sstream>
#include <string>
#include <vector>

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

    /**
     * Limit value to given range.
     *
     * Parameters
     * ----------
     * x : T const&
     *     Element to clip.
     * x_min : T const&
     *     Minimum value.
     * x_max : T const&
     *     Maximum value.
     *
     * Returns
     * -------
     * y : T
     *     Clipped element.
     */
    template <typename T>
    T clip(T const& x,
           T const& x_min,
           T const& x_max) {
        T y = static_cast<T>(0.0);
        if (x < x_min) {
            y = x_min;
        } else if (x > x_max) {
            y = x_max;
        } else {
            y = x;
        }

        return y;
    }

    /**
     * String representation of array.
     *
     * Parameters
     * ----------
     * x : std::vector<T> const&
     *
     * Returns
     * -------
     * msg : std::string
     *     Textual representation of array contents.
     */
    template <typename T>
    std::string print(std::vector<T> const& x) {
        std::stringstream msg;
        msg << "{";
        for(size_t i = 0; i < x.size() - 1; ++i) {
            msg << x[i] << ", ";
        }
        msg << x[x.size() - 1] << "}";

        return msg.str();
    }
}}
#endif //PYPELINE_UTIL_HPP
