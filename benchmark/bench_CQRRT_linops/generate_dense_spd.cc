#if defined(__APPLE__)
int main() {return 0;}
#else
#include "RandLAPACK/testing/rl_generate_dense_spd.hh"
int main(int argc, char* argv[]) {
    return RandLAPACK::testing::run_generate_dense_spd(argc, argv);
}
#endif
