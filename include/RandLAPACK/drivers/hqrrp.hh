


namespace HQRRP {

void dgeqpr(int64_t m, int64_t n, double *A, int64_t lda, int64_t *jpvt, double *tau);

void dgeqpr( int64_t * m, int64_t * n, double * A, int64_t * lda, int64_t * jpvt, double * tau,
         double * work, int64_t * lwork, int64_t * info );

int64_t hqrrp( int64_t m_A, int64_t n_A, double * buff_A, int64_t ldim_A,
        int64_t * buff_jpvt, double * buff_tau,
        int64_t nb_alg, int64_t pp, int64_t panel_pivoting );

} // end namespace HQRRP

