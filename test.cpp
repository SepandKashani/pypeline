// ############################################################################
// test.cpp
// ========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "eigen3/Eigen/Eigen"
#include "eigen3/Eigen/Sparse"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/embed.h"

#include "pypeline/ffs.hpp"
#include "pypeline/func.hpp"
#include "pypeline/linalg.hpp"
#include "pypeline/util.hpp"
#include "pypeline/transform.hpp"
#include "pypeline/bluebild.hpp"


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

void test_FFT() {
    namespace _ffs = pypeline::ffs;

    _ffs::FFT<float> FFT_f({5, 3}, 1, false, 1, _ffs::planning_effort::NONE);
    std::cout << FFT_f.__repr__() << std::endl;

    // Some dummy data for transforms
    pypeline::ArrayXX_t<double> A = pypeline::ArrayXX_t<double>::Constant(FFT_f.shape()[0], FFT_f.shape()[1], 1);

    // Fill FFT buffers
    Eigen::Map<pypeline::ArrayXX_t<std::complex<float>>> data_in(FFT_f.data_in(), A.rows(), A.cols());
    Eigen::Map<pypeline::ArrayXX_t<std::complex<float>>> data_out(FFT_f.data_out(), A.rows(), A.cols());
    data_in = A.cast<float>();

    std::cout << "Before fft()" << std::endl;
    std::cout << "IN_ADDR = " << &data_in(0) << std::endl;
    std::cout << data_in << std::endl;
    std::cout << "OUT_ADDR = " << &data_out(0) << std::endl;
    std::cout << data_out << std::endl;

    FFT_f.fft();

    std::cout << "After fft()" << std::endl;
    std::cout << "IN_ADDR = " << &data_in(0) << std::endl;
    std::cout << data_in << std::endl;
    std::cout << "OUT_ADDR = " << &data_out(0) << std::endl;
    std::cout << data_out << std::endl;

    FFT_f.ifft();

    std::cout << "After ifft()" << std::endl;
    std::cout << "IN_ADDR = " << &data_in(0) << std::endl;
    std::cout << data_in << std::endl;
    std::cout << "OUT_ADDR = " << &data_out(0) << std::endl;
    std::cout << data_out << std::endl;
    std::cout << std::endl;
}

// TODO: works with quick hack from Python for dirichlet kernel loading.
void test_FFS() {
    using TT = double;

    TT const T = M_PI;
    TT const T_c = M_E;
    size_t const N_FS = 15;
    size_t const N_samples = 18;

    pybind11::module util = pybind11::module::import("pypeline.util");
    pybind11::array_t<std::complex<TT>, pybind11::array::c_style | pybind11::array::forcecast> get_samples = util.attr("get_samples")();

    Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> diric_samples(reinterpret_cast<std::complex<TT>*>(get_samples.mutable_data()), 1, N_samples);
    Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> diric_FS(reinterpret_cast<std::complex<TT>*>(get_samples.mutable_data()) + N_samples, 1, N_samples);

    namespace _ffs = pypeline::ffs;

    _ffs::FFS<TT> FFS({1, N_samples}, 1, T, T_c, N_FS, false, 1, _ffs::planning_effort::NONE);
    std::cout << FFS.__repr__() << std::endl;

    // Fill FFS buffers
    Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> data_in(FFS.data_in(), 1, N_samples);
    Eigen::Map<pypeline::ArrayXX_t<std::complex<TT>>> data_out(FFS.data_out(), 1, N_samples);
    data_in = diric_samples;

    std::cout << "Before ffs()" << std::endl;
    std::cout << "IN_ADDR = " << &data_in(0) << std::endl;
    std::cout << data_in << std::endl;
    std::cout << "OUT_ADDR = " << &data_out(0) << std::endl;
    std::cout << data_out << std::endl;

    FFS.ffs();

    std::cout << "After ffs()" << std::endl;
    std::cout << "IN_ADDR = " << &data_in(0) << std::endl;
    std::cout << data_in << std::endl;
    std::cout << "OUT_ADDR = " << &data_out(0) << std::endl;
    std::cout << data_out << std::endl;

    FFS.iffs();

    std::cout << "After iffs()" << std::endl;
    std::cout << "IN_ADDR = " << &data_in(0) << std::endl;
    std::cout << data_in << std::endl;
    std::cout << "OUT_ADDR = " << &data_out(0) << std::endl;
    std::cout << data_out << std::endl;
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

void test_pol2cart() {
    namespace _transform = pypeline::transform;
    using T = double;


    size_t const N_colat = 3;
    pypeline::ArrayX_t<T> _colat = pypeline::ArrayX_t<T>::LinSpaced(N_colat, 0, 0.5 * M_PI);
    Eigen::Map<pypeline::ArrayXX_t<T>> colat(_colat.data(), N_colat, 1);

    size_t const N_lon = 4;
    pypeline::ArrayX_t<T> _lon = pypeline::ArrayX_t<T>::LinSpaced(N_lon, - 0.5 * M_PI, 0.5 * M_PI);
    Eigen::Map<pypeline::ArrayXX_t<T>> lon(_lon.data(), 1, N_lon);

    pypeline::ArrayXX_t<T> XYZ = _transform::pol2cart(colat, lon);
    std::cout << XYZ << std::endl;
    std::cout << std::endl;
}

void test_bluebild() {
    namespace _bluebild = pypeline::bluebild;
    using TT = double;

    TT const wl = 1.0;
    pypeline::ArrayX_t<TT> _grid_colat = pypeline::ArrayX_t<TT>::LinSpaced(5, M_PI / 10, M_PI / 2);
    Eigen::Map<pypeline::ArrayXX_t<TT>> grid_colat(_grid_colat.data(), 5, 1);
    pypeline::ArrayX_t<TT> _grid_lon = pypeline::ArrayX_t<TT>::LinSpaced(15, 0, M_PI / 10);
    Eigen::Map<pypeline::ArrayXX_t<TT>> grid_lon(_grid_lon.data(), 1, 15);
    TT const N_FS = 13;
    TT const T = 0.2 * M_PI;
    pypeline::ArrayXX_t<TT> R(3, 3);
    R << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    size_t const N_antenna = 4;
    size_t const N_eig = 2;
    size_t const N_threads = 1;
    pypeline::ffs::planning_effort effort = pypeline::ffs::planning_effort::NONE;

    _bluebild::FourierFieldSynthesizerBlock<TT>bb(wl, grid_colat, grid_lon, N_FS,
                                                  T, R, N_antenna, N_eig, N_threads, effort);


    size_t const N_beam = 2;
    pypeline::ArrayXX_t<TT> XYZ(N_antenna, 3);
    for (size_t i = 0; i < N_antenna * 3; ++i) {
        XYZ(i) = static_cast<TT>(i);
    }
    pypeline::ArrayXX_t<TT> V(N_beam, N_eig);
    for (size_t i = 0; i < N_beam * N_eig; ++i) {
        V(i) = static_cast<TT>(i);
    }
    pypeline::ArrayXX_t<TT> W_dense(N_antenna, N_beam);
    W_dense << 1, 0,
               1, 0,
               0, 1,
               0, 1;
    Eigen::SparseMatrix<TT, Eigen::RowMajor> W = W_dense.matrix().sparseView();

    // std::cout << XYZ << std::endl << std::endl;
    // std::cout << V << std::endl << std::endl;
    // std::cout << W << std::endl << std::endl;

    bb(V, XYZ, W);
    bb(V, XYZ, W);
}

int main() {
    pybind11::scoped_interpreter guard{};

    // // pypeline/ffs.hpp
    // test_next_fast_len();
    // test_ffs_sample();
    // test_FFT();
    // // test_FFS();

    // // pypeline/util.hpp
    // test_deg2rad();
    // test_clip();
    // test_print();

    // // pypeline/linalg.hpp
    // test_z_rot2angle();

    // // pypeline/func.hpp
    // test_Tukey();

    // // pypeline/transform.hpp
    // test_pol2cart();

    // pypeline/bluebild.hpp
    test_bluebild();

    return 0;
}
