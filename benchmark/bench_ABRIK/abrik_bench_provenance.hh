#ifndef ABRIK_BENCH_PROVENANCE_HH
#define ABRIK_BENCH_PROVENANCE_HH

#include <cstdlib>
#include <string>

/// Build provenance for ABRIK result files.
///
/// Every ABRIK CSV records what it ran on -- precision, input matrix, size, target rank --
/// but never recorded WHICH BINARY produced it. That gap is only visible when two eras get
/// compared: the rows are individually honest, the file looks complete, and nothing says the
/// numbers came from different builds.
///
/// It matters immediately for the adaptive growth study. The growth-1.25 cells exist to be
/// compared against the growth-2.0 cells, and that comparison spans two builds BY
/// CONSTRUCTION, because 1.25 cannot run until a build carries the adaptive_growth argument
/// that 2.0 predates. So the comparison the campaign is for is exactly the cross-build case,
/// and it needs the build named per cell rather than assumed constant.
///
/// The chain matches the one already used by the CQRRT campaigns, so a single MATLAB checker
/// works for both:
///
///   1. the install job writes the SHA to <root>/RandNLA-project/build/RANDLAPACK_COMMIT,
///      next to the binaries it describes rather than in a script;
///   2. each cell script cats that file and exports RANDLAPACK_GIT_COMMIT;
///   3. the binary echoes the env var into its comment header, here.
///
/// The binary reads an ENVIRONMENT VARIABLE rather than a compile-time define on purpose:
/// no CMake plumbing, no rebuild to re-stamp, and a broken chain shows up honestly as
/// "unknown" instead of silently reporting a stale SHA baked in at configure time.
///
/// Emitted as `# RANDLAPACK_GIT_COMMIT=<sha>` to match the checker's expected form, a 7-to-40
/// hex string anywhere in a leading '#' comment line.
inline std::string abrik_build_commit() {
    const char* v = std::getenv("RANDLAPACK_GIT_COMMIT");
    return (v && *v) ? std::string(v) : std::string("unknown");
}

#endif
