#pragma once

#include <RandLAPACK.hh>
#include <blas.hh>
#include <vector>
#include <algorithm>
#include <limits>
#include <map>
#include <exception>
#include <iterator>
#include <fstream>
#include <fast_matrix_market/fast_matrix_market.hpp>


namespace richol {


using symmetry_type = fast_matrix_market::symmetry_type;


template <typename ScalarType, typename OrdinalType>
struct VectorComponent {

    using scalar_t = ScalarType;
    using ordinal_t = OrdinalType;

    scalar_t  val = 0.0;
    ordinal_t ind = 0;

    VectorComponent() = default;
    VectorComponent(scalar_t s, ordinal_t o) : val(s), ind(o){ assert(ind >= 0); };
    VectorComponent(VectorComponent<scalar_t,ordinal_t>  &vc) : val(vc.val), ind(vc.ind) { assert(ind >= 0); };
    VectorComponent(VectorComponent<scalar_t,ordinal_t> &&vc) : val(vc.val), ind(vc.ind) { vc.ind = 0; vc.val = 0; };

};


template <typename ScalarType, typename OrdinalType>
struct SparseVec {

    using scalar_t  = ScalarType;
    using ordinal_t = OrdinalType;
    using comp_t    = VectorComponent<scalar_t, ordinal_t>;

    std::vector<comp_t> data{};

    SparseVec() = default;

    SparseVec( SparseVec<scalar_t, ordinal_t> &s ) : data(s.data) {}

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

    inline size_t size() { return data.size(); }

    inline bool empty() { return data.empty(); }

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
inline void xbapy(spvec_t x, scalar_t a, int64_t col_ind, std::vector<spvec_t> &csr_like) {
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
        auto [ vk, _k ] = v.leading_component();
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
    for (auto [vi, i] : v_data_copy) {
        // subtract off v_i v[i:] / v[k] from M[i]
        spvec_t  &row_i = M[i];
        scalar_t temp_i = vi / sqrt_vk;
        for (auto it = v_data_copy.begin() + ctr; it != v_data_copy.end(); ++it) {
            auto [vj, j] = (*it);
            scalar_t update_ij = -temp_i * (vj / sqrt_vk);
            row_i.push_back(j, update_ij);
        }
        ++ctr;
    }
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

    auto stopper = [zero_threshold](scalar_t vk, ordinal_t k) { return vk < zero_threshold;  }; // do NOT check agains abs(vk)
    auto downdater = [](spvec_t &v, std::vector<spvec_t> &M_intermediate) { downdaters::full(v, M_intermediate); };
    // TODO: try downdater = downdaters::full;

    ordinal_t k = abstract_cholesky(M, C, downdater, stopper, zero_threshold, handle_trailing_zero);
    return k;
}


} // end namespace richol
