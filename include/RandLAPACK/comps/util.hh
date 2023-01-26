#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

namespace RandLAPACK::comps::util {


/// Generates an identity matrix.
template <typename T>
void eye(
    int64_t m,
    int64_t n,
    std::vector<T>& A
);


/// Overwrites the diagonal entries of matrix S with those stored in s.
template <typename T>
void diag(
    int64_t m,
    int64_t n,
    const std::vector<T>& s,
    int64_t k,
    std::vector<T>& S
);

/// Captures k diagonal elements of A and stores them in buf.
template <typename T> 
void extract_diag(
    int64_t m,
    int64_t n,
    int64_t k, 
    const std::vector<T>& A,
    std::vector<T>& buf 
); 

/// Displays the first k diagonal elements.
template <typename T> 
void disp_diag(
    int64_t m,
    int64_t n,
    int64_t k, 
    const std::vector<T>& A 
);

/// Extracts the l-portion of the GETRF result, places 1's on the main diagonal.
/// Overwrites the passed-in matrix.
template <typename T> 
void get_L(
    int64_t m,
    int64_t n,
    std::vector<T>& L
);

/// Stores the upper-triangualr portion of A in U.
template <typename T> 
void get_U(
    int64_t m,
    int64_t n,
    const std::vector<T>& A,
    std::vector<T>& U
);

/// Positions columns of A in accordance with idx vector of length k.
/// idx array modified ONLY within the scope of this function.
template <typename T> 
void col_swap(
    int64_t m,
    int64_t n,
    int64_t k,
    std::vector<T>& A, 
    std::vector<int64_t> idx 
);

/// Checks if the given size is larger than available. If so, resizes the vector.
template <typename T> 
T* upsize(
    int64_t target_sz,
    std::vector<T>& A
);

/// Changes the number of rows of a column-major matrix.
template <typename T> 
T* row_resize(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t k
);

/// Dimensions m and n may change if we want the diagonal matrix of rank k < min(m, n).
/// In that case, it would be of size k by k.
template <typename T> 
void gen_mat_type(
    int64_t& m, // These may change
    int64_t& n,
    std::vector<T>& A,
    int64_t k, 
    int32_t seed,
    const std::tuple<int, T, bool>& type
);

/// Generates matrix with the following singular values:
/// sigma_i = 1 / (i + 1)^pow (first k * 0.2 sigmas = 1
/// Can either be a diagonal matrix, or a full one.
/// In later case, left and right singular vectors are randomly-generated 
/// and orthogonaized.
template <typename T> 
void gen_poly_mat(
    int64_t& m,
    int64_t& n,
    std::vector<T>& A,
    int64_t k,
    T cond, 
    bool diagon,
    int32_t seed
);

/// Generates matrix with the following singular values:
/// sigma_i = e^((i + 1) * -pow) (first k * 0.2 sigmas = 1
/// Can either be a diagonal matrix, or a full one.
/// In later case, left and right singular vectors are randomly-generated 
/// and orthogonaized.
template <typename T> 
void gen_exp_mat(
    int64_t& m,
    int64_t& n,
    std::vector<T>& A,
    int64_t k,
    T cond, 
    bool diagon,
    int32_t seed
);

/// Generates left and right singular vectors for the three matrix types above.
template <typename T> 
void gen_mat(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t k,
    std::vector<T>& S,
    int32_t seed
);

/// Find the condition number of a given matrix A.
template <typename T> 
T cond_num_check(
    int64_t m,
    int64_t n,
    const std::vector<T>& A,
    std::vector<T>& A_cpy,
    std::vector<T>& s,
    bool verbosity
);

/// Checks whether matrix A has orthonormal columns.
template <typename T> 
bool orthogonality_check(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::vector<T>& A,
    std::vector<T>& A_gram,
    bool verbosity
);

} // end namespace RandLAPACK::comps::rs
