#include "RandLAPACK.hh"
#include <RandBLAS.hh>
#include <fstream>
#include <vector>
#include "../../functions/misc/dm_util.hh"

int main() {
    int64_t n = 1138;
    double cond_num = 10.0;
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    std::string filename = "/home/mymel/data/CQRRT_linop_test_matrices/left_op/proper_spd_1138.mtx";
    RandLAPACK_demos::generate_spd_matrix_file<double>(filename, n, cond_num, state);

    std::cout << "Generated proper SPD matrix " << n << "x" << n << " with condition number " << cond_num << "\n";
    std::cout << "File: " << filename << "\n";

    return 0;
}
