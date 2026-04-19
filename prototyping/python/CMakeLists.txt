find_package(pybind11 REQUIRED)

pybind11_add_module(qrbbrp qrbbrp_bind.cc)
target_compile_features(qrbbrp PRIVATE cxx_std_20)
target_link_libraries(qrbbrp PRIVATE RandLAPACK)
message(STATUS "RandLAPACK: added qrbbrp Python module")
