#include "richol_core.hh"
#include <fstream>
#include <string>
#include <filesystem>
#include <vector>

using std::vector;


template <typename T>
RandBLAS::COOMatrix<T> read_negative_M_matrix(std::string &datadir) {
    auto A_coo = richol::coo_from_matrix_market<T>(datadir + "/p_rgh_matrix_A.mtx");
    std::for_each(A_coo.vals, A_coo.vals + A_coo.nnz, [](T &a) { a*= -1; return; } );
    T reg = 0.0;
    bool offdiag_of_A_is_nonpos = true;
    for (int64_t i = 0; (i < A_coo.nnz) && offdiag_of_A_is_nonpos; ++i) {
        if (A_coo.rows[i] == A_coo.cols[i]) {
            A_coo.vals[i] += reg;
        } else {
            offdiag_of_A_is_nonpos = A_coo.vals[i] <= 0;
        }
    }
    randblas_require(offdiag_of_A_is_nonpos);
    return A_coo;
}

template <typename T>
std::pair<vector<T>,vector<T>> setup_pcg_vecs(std::string &datadir, vector<int64_t> &perm) {
    vector<T> b0 = richol::vector_from_matrix_market<T>(datadir + "/p_rgh_source_b.mtx");
    int64_t n = static_cast<int64_t>(b0.size());
    vector<T> b(n);
    for (int64_t i = 0; i < n; ++i) { b[perm[i]] = -b0[i]; }
    vector<T> x0 = richol::vector_from_matrix_market<T>(datadir + "/p_rgh_psi_initial_x^{n}.mtx");
    vector<T> x(n);
    for (int64_t i = 0; i < n; ++i) { x[perm[i]] = x0[i]; }
    return {x, b};
}


int main(int argc, char** argv) {
    if(argc < 4) {
      std::cerr << "Usage: "<<argv[0]
                << " <data_dir> <out_dir> <seed> [--amd]\n";
      return 1;
    }
    std::string datadir = argv[1];
    std::string outbase = argv[2];
    int seed            = std::stoi(argv[3]);
    bool use_amd        = (argc > 4 && std::string(argv[4]) == "--amd");

    std::string outfull = outbase + ((use_amd) ? "/amd_true" : "/amd_false");

    // 0) ensure the output directory exists
    try {
        std::filesystem::create_directories(outfull);
    } catch(std::exception& e) {
        std::cerr << "Could not create output directory " << outfull << ": " << e.what() << "\n";
        return 1;
    }

    using T = double;
    using spvec = richol::SparseVec<T,int64_t>;
    using CSR_t = RandBLAS::CSRMatrix<T, int64_t>;

    // 1) read the matrix A in COO form, negate if necessary
    auto A_coo = read_negative_M_matrix<T>(datadir);
    int64_t n  = A_coo.n_rows; 

    // 2) optional AMD permutation
    vector<int64_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    if (use_amd) {
      richol::amd_permutation(A_coo.as_owning_csr(), perm);
      A_coo.symperm_inplace(perm.data());
    }

    // 3) build CSR and then upper-triangle CSR-like
    auto A_csr = A_coo.as_owning_csr();
    vector<spvec> a_csrlike;
    richol::csrlike_from_csr(A_csr, a_csrlike, /*keep=*/blas::Uplo::Upper);

    // 4) call randomized incomplete Cholesky
    vector<spvec> c_lower_csrlike;
    RandBLAS::RNGState state(seed);
    richol::clb21_rand_cholesky(a_csrlike, c_lower_csrlike, state, /*diag_adjust=*/true);
    CSR_t C_lower(n, n);
    richol::csr_from_csrlike(c_lower_csrlike, C_lower);

    // 5) write linear system data to disk.
    auto [x, b] = setup_pcg_vecs<T>(datadir, perm);

    std::ofstream ofs_x( outfull + "/x0.mtx" );
    std::ofstream ofs_b( outfull + "/b.mtx"  );
    std::ofstream ofs_C( outfull + "/richol_C_seed_" + std::to_string(seed) + ".mtx");
    std::ofstream ofs_A( outfull + "/A.mtx"  );

    std::string comment_x = "initial vector for linear system iterative solve";
    std::string comment_b = "linear system right-hand-side";
    std::string comment_C = "randomized incomplete Cholesky, seed = " + std::to_string(seed);
    std::string comment_A = "linear system matrix";

    richol::write_compressed_sparse(C_lower, ofs_C, comment_C);
    richol::write_compressed_sparse(A_csr,   ofs_A, comment_A);
    richol::write_vector(x, ofs_x, comment_x);
    richol::write_vector(b, ofs_b, comment_b);

    return 0;
}
