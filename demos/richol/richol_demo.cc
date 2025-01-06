
#include "richol.hh"

#include <iostream>

int main(int argc, char** argv) {
    using T = float;
    int64_t n = 8;
    int64_t nnz = 32;
    std::vector<T> vals{6, -1, -1, -1, -1, 6, -1, -1, -1, 6, -1, -1, -1, -1, 6, -1, -1, 6, -1, -1, -1, -1, 6, -1, -1, -1, 6, -1, -1, -1, -1, 6};
    std::vector<int64_t> rowptr{0, 4, 8, 12, 16, 20, 24, 28, 32};
    std::vector<int64_t> colidxs{0, 1, 2, 4, 0, 1, 3, 5, 0, 2, 3, 6, 1, 2, 3, 7, 0, 4, 5, 6, 1, 4, 5, 7, 2, 4, 6, 7, 3, 5, 6, 7};

    using spvec = richol::SparseVec<T, int64_t>;
    std::vector<spvec> sym;
    richol::sym_as_upper_tri_from_csr(n, rowptr.data(), colidxs.data(), vals.data(), sym);
    std::vector<spvec> C;
    auto k = richol::full_cholesky(sym, C);
    std::cout << "Exited with C of rank k = " << k << std::endl;

    return 0;
}