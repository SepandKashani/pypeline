// ############################################################################
// test.cpp
// ========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#include <iostream>

#include "pypeline/ffs.hpp"


void test_next_fast_len() {
    namespace _ffs = pypeline::ffs;

    size_t const target = 97;
    size_t const out = _ffs::next_fast_len(target);

    std::cout << target << std::endl;
    std::cout << out << std::endl;
}

int main() {
    pybind11::scoped_interpreter guard{};

    test_next_fast_len();

    return 0;
}
