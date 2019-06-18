// ############################################################################
// test.cpp
// ========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#include <cmath>
#include <iostream>
#include <vector>

#include "eigen3/Eigen/Eigen"

#include "pypeline/ffs.hpp"
#include "pypeline/func.hpp"
#include "pypeline/linalg.hpp"
#include "pypeline/util.hpp"


void test_next_fast_len() {
    namespace _ffs = pypeline::ffs;

    size_t const target = 97;
    size_t const out = _ffs::next_fast_len(target);

    std::cout << "next_fast_len(target=" << target << ") = " << out << std::endl;
    std::cout << std::endl;
}

void test_ffs_sample() {
    namespace _ffs = pypeline::ffs;

    double const T = 1;
    int const N_FS = 5;
    double const T_c = M_PI;

    int N_s = 8;
    std::cout << "ffs_sample(T=" << T << ", N_FS=" << N_FS << ", T_c=" << T_c << ", N_s=" << N_s << std::endl;
    std::cout << _ffs::ffs_sample<float>(T, N_FS, T_c, N_s) << std::endl;
    std::cout << _ffs::ffs_sample<double>(T, N_FS, T_c, N_s) << std::endl;
    std::cout << std::endl;

    N_s = 9;
    std::cout << "ffs_sample(T=" << T << ", N_FS=" << N_FS << ", T_c=" << T_c << ", N_s=" << N_s << std::endl;
    std::cout << _ffs::ffs_sample<float>(T, N_FS, T_c, N_s) << std::endl;
    std::cout << _ffs::ffs_sample<double>(T, N_FS, T_c, N_s) << std::endl;
    std::cout << std::endl;
}

void test_deg2rad() {
    namespace _util = pypeline::util;

    double const x = 90;
    std::cout << "deg2rad(x=" << x << ") = " << _util::deg2rad<float>(x) << std::endl;
    std::cout << "deg2rad(x=" << x << ") = " << _util::deg2rad<double>(x) << std::endl;
    std::cout << std::endl;
}

void test_clip() {
    namespace _util = pypeline::util;

    double const x_min = -1;
    double const x_max = 1;

    double x = -2;
    std::cout << "clip(x=" << x << ", x_min=" << x_min << ", x_max=" << x_max << ") = " << _util::clip<float>(x, x_min, x_max) << std::endl;
    std::cout << "clip(x=" << x << ", x_min=" << x_min << ", x_max=" << x_max << ") = " << _util::clip<double>(x, x_min, x_max) << std::endl;

    x = 2;
    std::cout << "clip(x=" << x << ", x_min=" << x_min << ", x_max=" << x_max << ") = " << _util::clip<float>(x, x_min, x_max) << std::endl;
    std::cout << "clip(x=" << x << ", x_min=" << x_min << ", x_max=" << x_max << ") = " << _util::clip<double>(x, x_min, x_max) << std::endl;

    x = -0.5;
    std::cout << "clip(x=" << x << ", x_min=" << x_min << ", x_max=" << x_max << ") = " << _util::clip<float>(x, x_min, x_max) << std::endl;
    std::cout << "clip(x=" << x << ", x_min=" << x_min << ", x_max=" << x_max << ") = " << _util::clip<double>(x, x_min, x_max) << std::endl;
    std::cout << std::endl;
}

void test_print() {
    namespace _util = pypeline::util;

    std::vector<size_t> x {1, 2, 3, 4};
    std::cout << _util::print(x) << std::endl;

    std::vector<int> y {-1, 0, 1, 2};
    std::cout << _util::print(y) << std::endl;

    std::vector<float> z {-0.5, 0.0, 0.5, 15};
    std::cout << _util::print(z) << std::endl;
    std::cout << std::endl;
}

void test_z_rot2angle() {
    namespace _linalg = pypeline::linalg;

    pypeline::ArrayXX_t<double> R(3, 3);
    R << 1, 0, 0,
         0, 1, 0,
         0, 0, 1;
    std::cout << "z_rot2angle(I_{3}) = " << _linalg::z_rot2angle(R.cast<float>()) << " [rad]" << std::endl;
    std::cout << "z_rot2angle(I_{3}) = " << _linalg::z_rot2angle(R) << " [rad]" << std::endl;

    R << 0, -1, 0,
         1,  0, 0,
         0,  0, 1;
    std::cout << "z_rot2angle([e_{2}, - e_{1}, e_{3}]) = " << _linalg::z_rot2angle(R.cast<float>()) << " [rad]" << std::endl;
    std::cout << "z_rot2angle([e_{2}, - e_{1}, e_{3}]) = " << _linalg::z_rot2angle(R) << " [rad]" << std::endl;
    std::cout << std::endl;
}

void test_Tukey() {
    namespace _func = pypeline::func;

    double const T = 1;
    double const beta = 0.5;
    double const alpha = 0.25;
    pypeline::ArrayX_t<double> _x = pypeline::ArrayX_t<double>::LinSpaced(25, 0, 1);
    Eigen::Map<pypeline::ArrayXX_t<double>> x(_x.data(), 5, 5);

    auto tukey_d = _func::Tukey<double>(T, beta, alpha);
    std::cout << tukey_d.__repr__() << "(x)" << std::endl;
    std::cout << tukey_d(x) << std::endl;
    std::cout << std::endl;

    auto tukey_f = _func::Tukey<float>(T, beta, alpha);
    std::cout << tukey_f.__repr__() << "(x)" << std::endl;
    std::cout << tukey_f(x.cast<float>()) << std::endl;
    std::cout << std::endl;
}

int main() {
    pybind11::scoped_interpreter guard{};

    // pypeline/ffs.hpp
    test_next_fast_len();
    test_ffs_sample();

    // pypeline/util.hpp
    test_deg2rad();
    test_clip();
    test_print();

    // pypeline/linalg.hpp
    test_z_rot2angle();

    // pypeline/func.hpp
    test_Tukey();

    return 0;
}
