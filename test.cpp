// ############################################################################
// test.cpp
// ========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#include <iostream>

#include "pypeline/ffs.hpp"
#include "pypeline/util.hpp"


void test_next_fast_len() {
    namespace _ffs = pypeline::ffs;

    size_t const target = 97;
    size_t const out = _ffs::next_fast_len(target);

    std::cout << target << std::endl;
    std::cout << out << std::endl;
}

void test_deg2rad() {
    namespace _util = pypeline::util;

    float x_f = 5;
    float y_f = _util::deg2rad(x_f);
    std::cout << x_f << ", " << y_f << std::endl;

    double x_d = 5;
    double y_d = _util::deg2rad(x_d);
    std::cout << x_d << ", " << y_d << std::endl;
}

int main() {
    pybind11::scoped_interpreter guard{};

    test_next_fast_len();
    test_deg2rad();

    return 0;
}
