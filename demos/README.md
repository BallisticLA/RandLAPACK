# RandLAPACK Demonstrations

The global scope of this project is yet to be determined.

## Dependencies

Demos automatically fetch these external libraries via CMake FetchContent:

- **Eigen** (https://eigen.tuxfamily.org/): Dense matrix library
- **Spectra** (https://spectralib.org/): Large-scale eigenvalue solver
- **fast_matrix_market**: Matrix file I/O (optional)

**No manual dependency installation required!** CMake will download them during configuration.

## Dependency Sharing with Benchmarks

These external dependencies are shared with the `benchmark/` project through CMake's FetchContent caching mechanism. When both projects declare the same dependency (e.g., `eigen_lib`), CMake automatically reuses the cached download, preventing redundant fetches.

