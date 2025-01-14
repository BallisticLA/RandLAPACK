#pragma once

#include <RandLAPACK.hh>
#include <blas.hh>
#include <vector>
#include <algorithm>
#include <limits>
#include <map>
#include <unordered_map>
#include <exception>
#include <iterator>
#include <fstream>
#include <numeric>       

#include <fast_matrix_market/fast_matrix_market.hpp>

#include <amd.h>


namespace richol {


using symmetry_type = fast_matrix_market::symmetry_type;
using RandBLAS::CSRMatrix;
using RandBLAS::sparse_data::reserve_csr;
using RandBLAS::sparse_data::reserve_coo;
using RandBLAS::CSCMatrix;
using RandBLAS::COOMatrix;
using RandBLAS::sparse_data::conversions::coo_to_csr;



template <typename ScalarType, typename OrdinalType>
struct VectorComponent {

    using scalar_t = ScalarType;
    using ordinal_t = OrdinalType;
    using self_t = VectorComponent<scalar_t, ordinal_t>;

    scalar_t  val = 0.0;
    ordinal_t ind = 0;

    VectorComponent() = default;
    VectorComponent(scalar_t s, ordinal_t o) : val(s), ind(o){ assert(ind >= 0); };
    VectorComponent(const self_t  &vc) : val(vc.val), ind(vc.ind) { assert(ind >= 0); };
    VectorComponent(self_t &&vc) : val(vc.val), ind(vc.ind) { vc.ind = 0; vc.val = 0; };

    self_t& operator=(const self_t &vc) { 
        val = vc.val;
        ind = vc.ind; 
        return *this;
    };

};


template <typename ScalarType, typename OrdinalType>
struct SparseVec {

    using scalar_t  = ScalarType;
    using ordinal_t = OrdinalType;
    using comp_t    = VectorComponent<scalar_t, ordinal_t>;

    std::vector<comp_t> data{};

    SparseVec() = default;

    SparseVec( const SparseVec<scalar_t, ordinal_t> &s ) : data(s.data) {}

    SparseVec( SparseVec<scalar_t, ordinal_t> &&s ) : data(std::move(s.data)) {}

    inline void push_back(ordinal_t ind, scalar_t val) { 
        data.push_back({val, ind});
    }

    void coallesce(scalar_t threshold = std::numeric_limits<scalar_t>::epsilon()) {
        std::map<ordinal_t, scalar_t> map;
        for (const comp_t &c : data)
            map[c.ind] += c.val;

        int64_t size = 0;
        for (const auto &pair : map) {
            // iterating over std::map automatically sorts by key in increasing order.
            if (std::abs(pair.second) >= threshold) {
                data[size].ind = pair.first;
                data[size].val = pair.second;
                ++size;
            }
        }
        data.resize(size);
        return;
    }

    ordinal_t max_ind(scalar_t threshold = std::numeric_limits<scalar_t>::epsilon()) const {
        ordinal_t ind = -1;
        for (const auto &c : data) {
            if (std::abs(c.val) >= threshold)
                ind = std::max(ind, c.ind);
        }
        return ind;
    }

    inline size_t size() const { return data.size(); }

    inline bool empty() const { return data.empty(); }

    inline void clear() { data.clear(); }

    inline comp_t leading_component() { return data[0]; }

};


template <typename ordinal_t1, typename ordinal_t2, typename vals_t, typename spvec_t>
void sym_as_upper_tri_from_csr( int64_t n, const ordinal_t1* rowptr, const ordinal_t2* colinds, const vals_t* vals, std::vector<spvec_t> &sym ) {
    using ordinal_t = typename spvec_t::ordinal_t;
    using scalar_t  = typename spvec_t::scalar_t;

    sym.clear();
    sym.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        auto &row  = sym[i];
        auto start = static_cast< int64_t >( rowptr[ i  ]  );
        auto stop  = static_cast< int64_t >( rowptr[ i+1 ] );
        for (int64_t nz_ind = start; nz_ind < stop; ++nz_ind) {
            auto j = static_cast<ordinal_t>(colinds[nz_ind]);
            if (i <= j) {
                auto val = static_cast<scalar_t>(vals[nz_ind]);
                row.push_back( j, val );
            }
        }
    }
    return;
}


template <typename ordinal_t1, typename ordinal_t2, typename vals_t, typename spvec_t>
void csr_from_csrlike(const std::vector<spvec_t> &csrlike, ordinal_t1* rowptr, ordinal_t2* colinds, vals_t* vals) {
    auto n = static_cast<int64_t>(csrlike.size());
    ordinal_t1 nzind = 0;
    for (int64_t i = 0; i < n; ++i) {
        const auto &row  = csrlike[i];
        rowptr[i] = nzind;
        for (const auto &[val, ind] : row.data) {
            colinds[nzind] = static_cast<ordinal_t2>(ind);
            vals   [nzind] = static_cast<vals_t>    (val);
            ++nzind;
        }
    }
    rowptr[n] = nzind;
    return;
}


template <typename T>
T epsilon() { return std::numeric_limits<T>::epsilon(); }


template <typename spvec_t, typename scalar_t = typename spvec_t::scalar_t>
size_t coallesce_spvecs(std::vector<spvec_t> &spvecs, scalar_t tol = epsilon<scalar_t>()) {
    size_t nnz = 0;
    for (spvec_t &row : spvecs) {
        row.coallesce(tol);
        nnz += row.size();
    }
    return nnz;
}


template <typename spvec_t, typename tol_t = typename spvec_t::scalar_t>
void write_square_matrix_market(
    std::vector<spvec_t> &csr_like, std::ostream &os, symmetry_type symtype, std::string comment = {}, tol_t tol = 0.0
) {
    size_t nz = coallesce_spvecs(csr_like, tol);
    std::vector<typename spvec_t::scalar_t > vals(nz);
    std::vector<typename spvec_t::ordinal_t> rows(nz);
    std::vector<typename spvec_t::ordinal_t> cols(nz);
    nz = 0;
    typename spvec_t::ordinal_t row_ind = 0;
    for (const spvec_t &row : csr_like) {
        for (const auto &comp : row.data) {
            vals[nz] = comp.val;
            rows[nz] = row_ind;
            cols[nz] = comp.ind;
            ++nz;
        }
        ++row_ind;
    }
    
    fast_matrix_market::matrix_market_header header(row_ind, row_ind);
    header.comment = comment;
    header.symmetry = symtype;

    fast_matrix_market::write_matrix_market_triplet(os, header, rows, cols, vals);
    return;
}


template <typename scalar_t, typename spvec_t>
inline void xbapy(spvec_t &x, scalar_t a, int64_t col_ind, std::vector<spvec_t> &csr_like) {
    for (const auto &xc : x.data) {
        csr_like[xc.ind].push_back(col_ind, xc.val/a);
    }
}


template <typename spvec_t, typename callable_t1, typename callable_t2>
typename spvec_t::ordinal_t abstract_cholesky(
    std::vector<spvec_t> &M,
    std::vector<spvec_t> &C,
    callable_t1 &downdate,
    callable_t2 &should_stop,
    typename spvec_t::scalar_t zero_threshold,
    bool handle_trailing_zero
) {
    using ordinal_t = typename spvec_t::ordinal_t;
    auto n = static_cast<ordinal_t>( M.size() );
    C.clear();
    C.resize(n);
    ordinal_t k = 0;
    while (k < n) {
        spvec_t &v = M[k];
        v.coallesce( zero_threshold );
        if (v.empty())
            break;
        const auto &[ vk, _k ] = v.leading_component();
        assert( k == _k );
        if (should_stop( vk, k ))
            break;
        xbapy( v, std::sqrt(vk), k, C );
        downdate( v, M );
        k += 1;
    }
    if (k == n-1 && handle_trailing_zero) {
        C[k].push_back(k, 1);
        k += 1;
    }
    return k;
}

namespace downdaters {

template <typename spvec_t>
void full(spvec_t &v, std::vector<spvec_t> &M) { 
    /**
     *   v is coallesced, v[:k] == 0, and v[k] > 0.
     *   Compute M -= outer(v, v)/v[k].
     */
    using comp_t    = typename spvec_t::comp_t;  
    using scalar_t  = typename spvec_t::scalar_t;
    using std::vector; using std::sqrt;
    
    vector<comp_t> v_data_copy(v.data);
    scalar_t sqrt_vk = sqrt(v_data_copy[0].val);
    int64_t ctr = 0;
    for (auto &[vi, i] : v_data_copy) {
        // subtract off v_i v[i:] / v[k] from M[i]
        spvec_t  &row_i = M[i];
        scalar_t temp_i = vi / sqrt_vk;
        for (auto it = v_data_copy.begin() + ctr; it != v_data_copy.end(); ++it) {
            auto &[vj, j] = (*it);
            scalar_t update_ij = -temp_i * (vj / sqrt_vk);
            row_i.push_back(j, update_ij);
        }
        ++ctr;
    }
    return;
}

template <typename scalar_t>
inline void scal(int64_t n, scalar_t alpha, scalar_t *x, int64_t incx) {
    blas::scal(n, alpha, x, incx);
}

template <typename spvec_t, typename state_t> 
void sample_clique_clb21(spvec_t &v, std::vector<spvec_t> &M, typename spvec_t::scalar_t p_sum_tol, bool diag_adjust, state_t &state) {
    /**
     *   v is coallesced, v[:k] == 0, and v[k] > 0.
     *   Compute M -= D, where E[D] = outer(v, v)/v[k].
     *   
     */
    using comp_t    = typename spvec_t::comp_t;
    using scalar_t  = typename spvec_t::scalar_t;
    using ordinal_t = typename spvec_t::ordinal_t; 
    using std::vector; using std::unordered_map;
    using std::min;    using std::max;
    ordinal_t ell;

    // split into the elimination index and trailing indices (neighbors)
    //
    // NOTE: there's no strict need to define a new "neighbors."
    //       it's safe to swap the leading compoinent of v.data to the end of v.data,
    //       then operate on the first num_neighbors entries of v.data in-place.
    const auto &[vk, k] = v.leading_component();
    vector<comp_t> neighbors(v.data.begin()+1, v.data.end());
    auto num_neighbors = static_cast<ordinal_t>(neighbors.size()); 
    for (ell = 0; ell < num_neighbors; ++ell)
        neighbors[ell].val *= -1;

    // sort neighbors in ascending order of edge weight; split into index and weight vectors.
    auto ascending_val = [](const comp_t &ca, const comp_t &cb){ return ca.val < cb.val; };
    std::sort(neighbors.begin(), neighbors.end(), ascending_val);
    vector<scalar_t>  neighbor_weights(num_neighbors);
    vector<ordinal_t> neighbor_indices(num_neighbors);
    unordered_map<ordinal_t, ordinal_t> ind_map;
    ell = 0;
    for (const comp_t &comp : neighbors) {
        auto &[val, ind] = comp;
        assert(val > 0);
        assert(ind > k);
        neighbor_weights[ell] = val;
        neighbor_indices[ell] = ind;
        ind_map[ind] = ell;
        ++ell;
    }

    // add the sampled spanning tree that approximates the clique in expectation
    auto num_neighbors_64t = static_cast<int64_t>(num_neighbors);
    int64_t sample_ind;
    vector<scalar_t> p;
    scalar_t trailing_weight = std::reduce(neighbor_weights.begin(), neighbor_weights.end());
    for (const auto &[abs_vi, i] : neighbors) {
        neighbor_weights[ind_map[i]] = static_cast<scalar_t>(0.0);
        p = neighbor_weights;
        scalar_t* p_buf = p.data();
        scalar_t  p_sum  = std::reduce(p_buf, p_buf + num_neighbors);
        if (p_sum <= p_sum_tol)
            break; // we should always hit this when i = neighbors.back().ind
        RandBLAS::weights_to_cdf(num_neighbors, p_buf);
        state = RandBLAS::sample_indices_iid(num_neighbors_64t, p_buf, 1, &sample_ind, state);
        ordinal_t j = neighbor_indices[sample_ind];
        scalar_t  w = (p_sum * abs_vi) / vk;
        M[i].push_back(i, w);
        M[min(i, j)].push_back(max(i, j), -w);
        M[j].push_back(j, w);
    }

    // subtract off the star centered at j.
    scalar_t scale = (diag_adjust) ? trailing_weight / vk : static_cast<scalar_t>(1);  
    for (auto it = v.data.begin() + 1; it != v.data.end(); ++it) {
        auto &[vi, i] = (*it);
        M[i].push_back(i, scale * vi);
    }
    M[k].clear();
    return;
}

} // end namespace richol::downdaters


template <typename spvec_t>
typename spvec_t::ordinal_t full_cholesky(
    std::vector<spvec_t>  M, // pass by value
    std::vector<spvec_t> &C, // pass by reference
    typename spvec_t::scalar_t zero_threshold = epsilon<typename spvec_t::scalar_t>(),
    bool handle_trailing_zero = true
) {
    using scalar_t  = typename spvec_t::scalar_t;
    using ordinal_t = typename spvec_t::ordinal_t;
    auto stopper = [zero_threshold](scalar_t vk, ordinal_t k) {  UNUSED(k); return vk < zero_threshold;  };
    // ^ do NOT check agains abs(vk)
    ordinal_t k = abstract_cholesky(M, C, downdaters::full<spvec_t>, stopper, zero_threshold, handle_trailing_zero);
    return k;
}

template <typename spvec_t, typename state_t>
typename spvec_t::ordinal_t clb21_rand_cholesky(
    std::vector<spvec_t>  M, // pass by value
    std::vector<spvec_t> &C, // pass by reference
    state_t &state,
    bool diag_adjust = false,
    typename spvec_t::scalar_t zero_threshold = epsilon<typename spvec_t::scalar_t>()
) {
    using scalar_t  = typename spvec_t::scalar_t;
    using ordinal_t = typename spvec_t::ordinal_t;
    auto n = static_cast<ordinal_t>(M.size());
    auto stopper = [zero_threshold, n, diag_adjust](scalar_t vk, ordinal_t k) { 
        ordinal_t processable_cols = (diag_adjust) ? n : n - 1;
        return (vk < zero_threshold) || (k == processable_cols); 
    };
    scalar_t p_sum_tol = zero_threshold;
    auto downdater = [p_sum_tol, diag_adjust, &state](spvec_t &v, std::vector<spvec_t> &M_work) { 
        downdaters::sample_clique_clb21(v, M_work, p_sum_tol, diag_adjust, state);
    };
    ordinal_t k = abstract_cholesky(M, C, downdater, stopper, zero_threshold, true);
    return k;
}


template <typename spvec_t>
std::vector<spvec_t> lift2lap(const std::vector<spvec_t> &sym) {
    std::vector<spvec_t> sym_lift(sym);
    using scalar_t  = typename spvec_t::scalar_t;
    using ordinal_t = typename spvec_t::ordinal_t;
    auto n = static_cast<ordinal_t>(sym.size());
    // set d = sym * vector-of-ones.
    std::vector<scalar_t> d(n, 0);
    for (ordinal_t i = 0; i < n; ++i) {
        auto &vec = sym_lift[i];
        vec.coallesce((scalar_t) 0);
        for (const auto &[val, ind] : vec.data) {
            d[i] += val;
            if (ind != i) {
                d[ind] += val;
            }
        }
    }
    // extend sym_lift = [sym  ,   -d ]
    //                   [ -d' , sum(d) ]
    for (ordinal_t i = 0; i < n; ++i) {
        auto &vec = sym_lift[i];
        vec.push_back(n, -d[i]);
    }
    scalar_t sum_d = std::reduce(d.begin(), d.end());
    sym_lift.resize(n+1);
    sym_lift[n].push_back(sum_d, n);
    return sym_lift;
}


template <typename scalar_t, RandBLAS::SignedInteger sint_t = int64_t>
CSRMatrix<scalar_t, sint_t> laplacian_from_matrix_market(std::string fn, scalar_t reg) {
    int64_t n, n_ = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<double> vals{};

    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n, n_, rows,  cols, vals
    );
    randblas_require(n == n_); // we must be square.

    // Convert adjacency matrix to COO format Laplacian
    int64_t m = vals.size();
    int64_t nnz = m + n;
    COOMatrix<double> coo(n, n);
    RandBLAS::sparse_data::reserve_coo(nnz, coo);
    std::vector<double> diagvec(n, reg);
    auto diag = diagvec.data();
    for (int64_t i = 0; i < m; ++i) {
        coo.rows[i] = rows[i];
        coo.cols[i] = cols[i];
        double v = vals[i];
        randblas_require(v >= 0);
        coo.vals[i] = -v;
        diag[rows[i]] += v;
    }
    for (int64_t i = 0; i < n; ++i) {
        coo.vals[m+i] = diag[i];
        coo.rows[m+i] = i;
        coo.cols[m+i] = i;
    }
    // convert COO format Laplacian to CSR format, using RandBLAS.
    CSRMatrix<double> csr(n, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(coo, csr);
    return csr;
}


template <typename scalar_t, RandBLAS::SignedInteger sint_t = int64_t>
void permuted(const CSRMatrix<scalar_t, sint_t> &A, const std::vector<sint_t> &perm, CSRMatrix<scalar_t, sint_t> &out) {
    auto  n   = A.n_rows;
    COOMatrix<scalar_t, sint_t> coo(n, n);
    reserve_coo(A.nnz, coo);
    std::vector<sint_t> invperm(n);
    for (sint_t k = 0; k < n; ++k) {
        invperm[perm[k]] = k;
    }
    sint_t inew, jnew, p, ctr = 0;
    for (jnew = 0; jnew < n; ++jnew) {
        auto j = perm[jnew];
        for (p = A.rowptr[j]; p < A.rowptr[j+1]; ++p) {
            inew = invperm[A.colidxs[p]];
            coo.rows[ctr] = inew;
            coo.cols[ctr] = jnew;
            coo.vals[ctr] = A.vals[p];
            ++ctr;
        }
    }
    coo_to_csr(coo, out);
    return;
}



} // end namespace richol
