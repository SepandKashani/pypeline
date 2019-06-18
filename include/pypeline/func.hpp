// ############################################################################
// func.hpp
// ========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

/*
 * Special 1D functions.
 */

#ifndef PYPELINE_FUNC_HPP
#define PYPELINE_FUNC_HPP

#include <cmath>
#include <stdexcept>
#include <sstream>
#include <string>
#include <type_traits>

#include "eigen3/Eigen/Eigen"

#include "pypeline/types.hpp"

namespace pypeline { namespace func {
    /**
     * Parameterized Tukey function.
     *
     * Notes
     * -----
     * The Tukey function is defined as:
     *
     * .. math::
     *
     *    \text{Tukey}(T, \beta, \alpha)(\varphi): \mathbb{R} & \to [0, 1] \\
     *    \varphi & \to
     *    \begin{cases}
     *        % LINE 1
     *        \sin^{2} \left( \frac{\pi}{T \alpha}
     *                 \left[ \frac{T}{2} - \beta + \varphi \right] \right) &
     *        0 \le \frac{T}{2} - \beta + \varphi < \frac{T \alpha}{2} \\
     *        % LINE 2
     *        1 &
     *        \frac{T \alpha}{2} \le \frac{T}{2} - \beta +
     *        \varphi \le T - \frac{T \alpha}{2} \\
     *        % LINE 3
     *        \sin^{2} \left( \frac{\pi}{T \alpha}
     *                 \left[ \frac{T}{2} + \beta - \varphi \right] \right) &
     *        T - \frac{T \alpha}{2} < \frac{T}{2} - \beta + \varphi \le T \\
     *        % LINE 4
     *        0 &
     *        \text{otherwise.}
     *    \end{cases}
     */
    template <typename TT>
    class Tukey {
        public:
            /*
             * Parameters
             * ----------
             * T : TT const&
             *     Function support.
             * beta : TT const&
             *     Function mid-point.
             * alpha : TT const&
             *     Decay-rate in [0, 1].
             */
            Tukey(TT const& T, TT const& beta, TT const& alpha):
                m_T(T), m_beta(beta), m_alpha(alpha) {
                if (T <= 0) {
                    std::string msg = "Parameter[T] must be positive.";
                    throw std::runtime_error(msg);
                }
                if (!((0 <= alpha) && (alpha <= 1))) {
                    std::string msg = "Parameter[alpha] must lie in [0, 1].";
                    throw std::runtime_error(msg);
                }
            }

            /*
             * Sample the Tukey(T, beta, alpha) function.
             *
             * Parameters
             * ----------
             * x : Eigen::ArrayBase<Derived> const&
             *     (N_height, N_width) sample points.
             *
             * Returns
             * -------
             * amplitude : pypeline::ArrayXX_t<Derived::Scalar>
             *     (N_height, N_width) Tukey values at sample points.
             */
            template <typename Derived>
            pypeline::ArrayXX_t<typename Derived::Scalar> operator()(Eigen::ArrayBase<Derived> const& x) {
                using T_t = typename Derived::Scalar;
                static_assert(std::is_same<TT, T_t>::value,
                              "Type[TT] and Type[Derived::Scalar] must be identical.");


                pypeline::ArrayXX_t<TT> y = x - m_beta + (m_T / 2);
                pypeline::ArrayXX_t<TT> amplitude(y.rows(), y.cols());
                amplitude.setZero();

                bool const alpha_0 = (m_alpha <= 1e-6);
                TT const lim_left = m_T * m_alpha / 2;
                TT const lim_right = m_T - (m_T * m_alpha / 2);

                for (size_t i = 0; i < y.size(); ++i) {
                    TT const _y = y(i);
                    bool const ramp_up = (0 <= _y) && (_y < lim_left);
                    bool const body = (lim_left <= _y) && (_y <= lim_right);
                    bool const ramp_down = (lim_right < _y) && (_y <= m_T);

                    if (body) {
                        amplitude(i) = static_cast<TT>(1.0);
                    } else if (! alpha_0) {
                        if (ramp_up) {
                            amplitude(i) = std::sin(M_PI / (m_T * m_alpha) * _y);
                        } else {
                            amplitude(i) = std::sin(M_PI / (m_T * m_alpha) * (m_T - _y));
                        }
                    }
                }

                amplitude = amplitude.square();
                return amplitude;
            }

            std::string __repr__() {
                std::stringstream msg;
                msg << "Tukey("
                    << "T=" << std::to_string(m_T) << ", "
                    << "beta=" << std::to_string(m_beta) << ", "
                    << "alpha=" << std::to_string(m_alpha) << ")";

                return msg.str();
            }

        private:
            TT const m_T = 0;
            TT const m_beta = 0;
            TT const m_alpha = 0;
    };
}}
#endif //PYPELINE_FUNC_HPP
