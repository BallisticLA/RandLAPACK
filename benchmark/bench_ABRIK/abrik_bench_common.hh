#ifndef ABRIK_BENCH_COMMON_HH
#define ABRIK_BENCH_COMMON_HH

// Shared by the ABRIK benchmark drivers: the build provenance stamp, the timestamped
// CSV file, and the precision dispatch of main().

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

/// The commit the binary was built from, as the launching script exports it in the
/// environment variable RANDLAPACK_GIT_COMMIT; "unknown" when it is absent. Every driver
/// writes it as `# RANDLAPACK_GIT_COMMIT=<sha>` in its CSV header, so a result file names
/// the build it came from. An environment variable rather than a compile-time define: no
/// CMake plumbing, no rebuild to re-stamp, and a broken chain shows up as "unknown"
/// instead of a stale SHA baked in at configure time.
inline std::string abrik_build_commit() {
    const char* v = std::getenv("RANDLAPACK_GIT_COMMIT");
    return (v && *v) ? std::string(v) : std::string("unknown");
}

/// Opens <output_dir>/<YYYYmmdd_HHMMSS>_<name>.csv for writing (no directory prefix when
/// output_dir is "."), the naming the plotting scripts look for, and returns the path.
/// The caller checks the stream: a missing directory leaves it closed.
inline std::string abrik_open_csv(const std::string& output_dir, const std::string& name,
                                  std::ofstream& out) {
    std::time_t now = std::time(nullptr);
    char prefix[20];
    std::strftime(prefix, sizeof(prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));
    std::string filename = std::string(prefix) + name + ".csv";
    std::string path = (output_dir != ".") ? output_dir + "/" + filename : filename;
    out.open(path);
    if (!out) std::cerr << "Error: cannot open " << path << " for writing\n";
    return path;
}

/// Prints the usage line and returns the exit status of an argument error.
inline int abrik_usage(const char* argv0, const char* usage) {
    std::cerr << "Usage: " << argv0 << " " << usage << "\n";
    return 2;
}

/// The main() shared by the drivers: dispatch on argv[1], "double" or "float" ("single"
/// is accepted as a synonym). The run functions return the exit status.
inline int abrik_bench_main(int argc, char* argv[], const char* usage,
                            int (*run_double)(int, char**), int (*run_float)(int, char**)) {
    if (argc < 2) return abrik_usage(argv[0], usage);
    std::string precision = argv[1];
    if (precision == "double") return run_double(argc, argv);
    if (precision == "float" || precision == "single") return run_float(argc, argv);
    std::cerr << "Error: precision must be 'double' or 'float', got '" << precision << "'\n";
    return abrik_usage(argv[0], usage);
}

#endif
