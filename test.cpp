// ############################################################################
// test.cpp
// ========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#include <cmath>
#include <iostream>

#include "eigen3/Eigen/Eigen"

#include "pypeline/ffs.hpp"
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

int main() {
    pybind11::scoped_interpreter guard{};

    test_next_fast_len();
    test_ffs_sample();
    test_deg2rad();

    return 0;
}
