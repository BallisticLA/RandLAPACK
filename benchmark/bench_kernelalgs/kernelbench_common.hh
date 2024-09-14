#pragma once

#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#include <chrono>
#include <unordered_map>
#include <iomanip> 
#include <limits> 
#include <numbers>

#include <iostream>
#include <fstream>

#include <fast_matrix_market/fast_matrix_market.hpp>

using std_clock = std::chrono::high_resolution_clock;
using timepoint_t = std::chrono::time_point<std_clock>;
using std::chrono::duration_cast;
using std::chrono::microseconds;

using RandBLAS::RNGState;
using RandLAPACK::rp_cholesky;
using lapack::gesdd;
using lapack::Job;
using std::vector;

double sec_elapsed(timepoint_t tp0, timepoint_t tp1) {
    return ((double) duration_cast<microseconds>(tp1 - tp0).count())/1e6;
}

template <typename T>
void transpose_colmajor(
    int64_t m, int64_t n, const T* A, int64_t lda, T* AT, int64_t ldat
) {
    for(int i = 0; i < n; ++i)
        blas::copy(m, &A[i * lda], 1, &AT[i], ldat);
}


struct array_matrix {
    int64_t nrows = 0, ncols = 0;
    std::vector<double> vals;
};

struct KRR_data {
    array_matrix X_train;
    array_matrix Y_train;
    array_matrix X_test;
    array_matrix Y_test;
};

void standardize(KRR_data &krrd) {
    randblas_require(krrd.X_train.nrows == krrd.X_test.nrows);
    using T = double;
    int64_t d = krrd.X_train.nrows;
    std::vector<T> mu(d, 0.0);
    std::vector<T> sigma(d, 0.0);
    RandLAPACK::standardize_dataset(
        d, krrd.X_train.ncols, krrd.X_train.vals.data(), mu.data(), sigma.data(), false
    );
    RandLAPACK::standardize_dataset(
        d, krrd.X_test.ncols, krrd.X_test.vals.data(), mu.data(), sigma.data(), true
    );
    return;
}

array_matrix mmread_file(std::string fn, bool transpose = true) {
    array_matrix mat{};
    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_array(
        file_stream, mat.nrows, mat.ncols, mat.vals, fast_matrix_market::col_major
    );
    if (transpose) {
        array_matrix tmat{};
        tmat.nrows = mat.ncols;
        tmat.ncols = mat.nrows;
        tmat.vals.resize(mat.vals.size(), 0.0);
        transpose_colmajor(
            mat.nrows, mat.ncols, mat.vals.data(), mat.nrows, tmat.vals.data(), tmat.nrows
        );
        return tmat;
    } else {
        return mat;   
    }
}

KRR_data mmread_krr_data_dir(std::string dn) {
    // mmread_file calls below always apply a transpose; might need to skip transposition for some
    // datasets.
    KRR_data data{};
    data.X_train = mmread_file(dn + "/Xtr.mm");
    data.Y_train = mmread_file(dn + "/Ytr.mm");
    data.X_test  = mmread_file(dn + "/Xts.mm");
    data.Y_test  = mmread_file(dn + "/Yts.mm");
    standardize(data);
    return data;
}

namespace memprof {
/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)
#include <psapi.h>
#include <windows.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) ||                 \
    (defined(__APPLE__) && defined(__MACH__))
#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) ||                              \
    (defined(__sun__) || defined(__sun) ||                                     \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) ||              \
    defined(__gnu_linux__)
#include <stdio.h>

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
inline size_t getPeakRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) ||                              \
    (defined(__sun__) || defined(__sun) ||                                     \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
  /* AIX and Solaris ------------------------------------------ */
  struct psinfo psinfo;
  int fd = -1;
  if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
    return (size_t)0L; /* Can't open? */
  if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
    close(fd);
    return (size_t)0L; /* Can't read? */
  }
  close(fd);
  return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) ||                 \
    (defined(__APPLE__) && defined(__MACH__))
  /* BSD, Linux, and OSX -------------------------------------- */
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
  return (size_t)rusage.ru_maxrss;
#else
  return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
  /* Unknown OS ----------------------------------------------- */
  return (size_t)0L; /* Unsupported. */
#endif
}

/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
inline size_t getCurrentRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
  /* OSX ------------------------------------------------------ */
  struct mach_task_basic_info info;
  mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                &infoCount) != KERN_SUCCESS)
    return (size_t)0L; /* Can't access? */
  return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) ||              \
    defined(__gnu_linux__)
  /* Linux ---------------------------------------------------- */
  long rss = 0L;
  FILE *fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return (size_t)0L; /* Can't open? */
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
    fclose(fp);
    return (size_t)0L; /* Can't read? */
  }
  fclose(fp);
  return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
  /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
  return (size_t)0L; /* Unsupported. */
#endif
}

// inline void log_pages() {
//     static size_t pagesize = sysconf(_SC_PAGESIZE);
//     int64_t bytes = getCurrentRSS();
//     assert((bytes % pagesize) == 0);
//     size_t pages = bytes / pagesize;
//     std::cout << "page size: " << pagesize << "\t";
//     std::cout << "bytes: " << bytes << "\t";
//     std::cout << "pages: " << pages << std::endl;
//     return;
// }

inline void log_pages(std::ostream &stream) {
    static size_t pagesize = sysconf(_SC_PAGESIZE);
    int64_t bytes = getCurrentRSS();
    assert((bytes % pagesize) == 0);
    size_t pages = bytes / pagesize;
    stream << "page size: " << pagesize << "\t";
    stream << "bytes: " << bytes << "\t";
    stream << "pages: " << pages << std::endl;
    return;
}

inline void log_memory_gb(std::ostream &stream) {
    int64_t bytes = getCurrentRSS();
    double gb = ((double) bytes) / ((double) std::pow(1024,3));
    stream << " Memory (GB)  : " << gb << "\n";
    return;
}

}