To build, run the following from this directory.

```bash

cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=<randlapack-install-prefix> \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir)

cmake --build build --target qrbbrp
```
