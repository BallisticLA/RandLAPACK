#if defined(__APPLE__)
int main() {return 0;}
#else
#include "RandLAPACK/testing/rl_generate_sparse_spd.hh"
int main(int argc, char* argv[]) {
    return RandLAPACK::testing::run_generate_sparse_spd(argc, argv);
}
#endif
