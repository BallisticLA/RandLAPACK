#include <RandLAPACK.hh>

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>

void run_pcgls_ex(int n, int m)
{
    std::vector<double> A(m * n);
    for (uint64_t i = 0; i < A.size(); ++i) {
        A[i] = ((double)i + 1.0) / m;
    }
    std::vector<double> b(m);
    for (uint64_t i = 0; i < b.size(); ++i) {
        b[i] = 1.0 / ((double) (i+1));
    }
    std::vector<double> c(n, 0.0);
    std::vector<double> M(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        M[i + n*i] = 1.0;
    }
    std::vector<double> x0(n, 0.0);
    std::vector<double> x(n, 0.0);
    std::vector<double> y(m, 0.0);
    std::vector<double> resid_vec(10*n, -1.0);

    double delta = 0.1;
    double tol = 1e-8;

    RandLAPACK::pcg(m, n, A.data(), m, b.data(), c.data(), delta,
        resid_vec, tol, n, M.data(), n, x0.data(), x.data(), y.data());

    for (double res: resid_vec)
    {
        if (res < 0) {
            break;
        }
        std::cout << res << "\n";
    }
}


int main(int, char **argv)
{
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);

    run_pcgls_ex(n, m);

    return 0;
}
