#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <RandBLAS.hh>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <numeric>

#include "RandLAPACK/drivers/rl_bqrrp_gpu.hh"

using GPUSubroutines = RandLAPACK::BQRRPGPUSubroutines;
using namespace std::chrono;

template <typename T>
struct BQRRPBenchData {
    int64_t row;
    int64_t col;
    int64_t rank;

    std::vector<T> A;
    T* A_sk;
    // Buffers for the GPU data
    T* A_device;
    T* A_sk_device;
    T* tau_device;
    int64_t* J_device;

    T* R_device;
    T* D_device;

    BQRRPBenchData(int64_t m, int64_t n) :
    A(m * n, 0.0)
    {
        row = m;
        col = n;
        cudaMalloc(&A_device,    m * n * sizeof(T));
        cudaMalloc(&tau_device,  n *     sizeof(T));
        cudaMalloc(&J_device,    n *     sizeof(int64_t));
        cudaMalloc(&R_device,    n * n * sizeof(T));
        cudaMalloc(&D_device,    n *     sizeof(T));
    }

    ~BQRRPBenchData() {
        cudaFree(A_device);
        cudaFree(tau_device);
        cudaFree(J_device);
        cudaFree(R_device);
        cudaFree(D_device);
    }
};

template <typename T, typename RNG>
static void data_regen(
                        RandLAPACK::gen::mat_gen_info<T> m_info,
                        BQRRPBenchData<T> &all_data,
                        RandBLAS::RNGState<RNG> &state) {

    auto state_const = state;
    auto m = m_info.rows;
    auto n = m_info.cols;

    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state_const);
    cudaMemset(all_data.J_device, 0.0, n);
    cudaMemset(all_data.tau_device, 0.0, n);
}

template <typename T, typename RNG>
static void bench_BQRRP(
    bool profile_runtime,
    bool run_qrf,
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t m,
    int64_t n,
    int64_t block_size,
    BQRRPBenchData<T> &all_data,
    RandBLAS::RNGState<RNG> state,
    std::string* output_filename_breakdown_QRF,
    std::string* output_filename_breakdown_CholQR,
    std::string* output_filename_speed) {

    T d_factor = 1.0;
    auto state_const = state;
    int64_t d = d_factor * block_size;

    // BQRRP with QRF
    // Skethcing in an sampling regime
    cudaMalloc(&all_data.A_sk_device, d * n * sizeof(T));
    all_data.A_sk = new T[d * n]();
    T* S          = new T[d * m]();

    RandBLAS::DenseDist D(d, m);
    RandBLAS::fill_dense(D, S, state_const);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, 1.0, S, d, all_data.A.data(), m, 0.0, all_data.A_sk, d);
    cudaMemcpy(all_data.A_sk_device, all_data.A_sk, d * n * sizeof(double), cudaMemcpyHostToDevice);
    RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU_QRF(profile_runtime, block_size);
    BQRRP_GPU_QRF.qr_tall = GPUSubroutines::QRTall::geqrf;
    auto start_bqrrp_qrf = std::chrono::steady_clock::now();
    BQRRP_GPU_QRF.call(m, n, all_data.A_device, m, all_data.A_sk_device, d, all_data.tau_device, all_data.J_device);
    auto stop_bqrrp_qrf = std::chrono::steady_clock::now();
    auto diff_bqrrp_qrf = std::chrono::duration_cast<std::chrono::microseconds>(stop_bqrrp_qrf - start_bqrrp_qrf).count();
    data_regen(m_info, all_data, state);
    cudaFree(all_data.A_sk_device);
    delete[] all_data.A_sk;

    if(profile_runtime) {
        std::ofstream file(*output_filename_breakdown_QRF, std::ios::app);
        std::copy(BQRRP_GPU_QRF.times.data(), BQRRP_GPU_QRF.times.data() + 15, std::ostream_iterator<T>(file, ", "));
        file << "\n";
    }

    // BQRRP with CholQR
    // Skethcing in an sampling regime
    cudaMalloc(&all_data.A_sk_device, d * n * sizeof(T));
    all_data.A_sk = new T[d * n]();
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, 1.0, S, d, all_data.A.data(), m, 0.0, all_data.A_sk, d);
    delete[] S;
    cudaMemcpy(all_data.A_sk_device, all_data.A_sk, d * n * sizeof(double), cudaMemcpyHostToDevice);
    RandLAPACK::BQRRP_GPU<double, r123::Philox4x32> BQRRP_GPU_CholQR(profile_runtime, block_size);
    BQRRP_GPU_CholQR.qr_tall = GPUSubroutines::QRTall::cholqr;
    auto start_bqrrp_cholqr = std::chrono::steady_clock::now();
    BQRRP_GPU_CholQR.call(m, n, all_data.A_device, m, all_data.A_sk_device, d, all_data.tau_device, all_data.J_device);
    auto stop_bqrrp_cholqr = std::chrono::steady_clock::now();
    auto diff_bqrrp_cholqr = std::chrono::duration_cast<std::chrono::microseconds>(stop_bqrrp_cholqr - start_bqrrp_cholqr).count();
    data_regen(m_info, all_data, state);
    cudaFree(all_data.A_sk_device);
    delete[] all_data.A_sk;

    if(profile_runtime) {
        std::ofstream file(*output_filename_breakdown_CholQR, std::ios::app);
        std::copy(BQRRP_GPU_CholQR.times.data(), BQRRP_GPU_CholQR.times.data() + 15, std::ostream_iterator<T>(file, ", "));
        file << "\n";
    }

    // Optional QRF
    long diff_qrf = 0;
    if (run_qrf) {
        lapack::Queue lapack_queue(0);
        using lapack::device_info_int;
        device_info_int* d_info = blas::device_malloc< device_info_int >( 1, lapack_queue );
        char* d_work_geqrf;
        char* h_work_geqrf;
        size_t d_size_geqrf, h_size_geqrf;

        auto start_qrf = std::chrono::steady_clock::now();
        lapack::geqrf_work_size_bytes(m, n, all_data.A_device, m, &d_size_geqrf, &h_size_geqrf, lapack_queue);
        d_work_geqrf = blas::device_malloc< char >( d_size_geqrf, lapack_queue );
        std::vector<char> h_work_geqrf_vector( h_size_geqrf );
        h_work_geqrf = h_work_geqrf_vector.data();
        lapack::geqrf(m, n, all_data.A_device, m, all_data.tau_device, d_work_geqrf, d_size_geqrf, h_work_geqrf, h_size_geqrf, d_info, lapack_queue);
        lapack_queue.sync();
        auto stop_qrf  = std::chrono::steady_clock::now();
        diff_qrf  = std::chrono::duration_cast<std::chrono::microseconds>(stop_qrf  - start_qrf).count();
        std::cout << " QRF TIME (MS) = " << diff_qrf << "\n";
    }

    std::cout << "  BLOCK SIZE = " << block_size << " BQRRP+QRF TIME (MS) = " << diff_bqrrp_qrf << " BQRRP+CholQR TIME (MS) = " << diff_bqrrp_cholqr << "\n";
    std::ofstream file(*output_filename_speed, std::ios::app);
    file << diff_bqrrp_qrf << "  " << diff_bqrrp_cholqr << "  " << diff_qrf << "\n";
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Error before bench_bqrrp returned. " << cudaGetErrorString(ierr))
        abort();
    }
}

static void run_block_size_sweep(
    int64_t m,
    int64_t n,
    std::vector<int64_t> b_sz,
    bool profile_runtime,
    bool run_qrf
){
    // Get a string representation of the block size vector
    std::string b_sz_string = std::accumulate(b_sz.begin(), b_sz.end(), std::string(),
                                [](const std::string& a, int b) {
                                    return a.empty() ? std::to_string(b) : a + "," + std::to_string(b);
                                });

    auto state           = RandBLAS::RNGState();

    BQRRPBenchData<double> all_data(m, n);
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
    cudaMemcpy(all_data.A_device, all_data.A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);

    std::string* file_name_1 = nullptr;
    std::string* file_name_2 = nullptr;
    if (profile_runtime) {
        file_name_1 = new std::string("_BQRRP_GPU_runtime_breakdown_qrf_num_info_lines_" + std::to_string(6) + ".txt");
        file_name_2 = new std::string("_BQRRP_GPU_runtime_breakdown_cholqr_num_info_lines_" + std::to_string(6) + ".txt");

        std::ofstream file1(*file_name_1, std::ios::out | std::ios::app);
        std::ofstream file2(*file_name_2, std::ios::out | std::ios::app);

        file1 << "Description: Results from the BQRRP GPU runtime breakdown benchmark, recording the time it takes to perform every subroutine in BQRRP."
                "\nFile format: 15 data columns, each corresponding to a given BQRRP subroutine: preallocation_t_dur, qrcp_main_t_dur, copy_A_sk_t_dur, qrcp_piv_t_dur, copy_A_t_dur, piv_A_t_dur, copy_J_t_dur, updating_J_t_dur, preconditioning_t_dur, qr_tall_t_dur, q_reconstruction_t_dur, apply_transq_t_dur, sample_update_t_dur, t_rest, total_t_dur"
                "               rows correspond to BQRRP runs with block sizes varying in a way unique for a particular run."
                "\nInput type:"       + std::to_string(m_info.m_type) +
                "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
                "\nAdditional parameters: Tall QR subroutine cholqr BQRRP block sizes: " + b_sz_string +
                "\n";
        file1.flush();

        file2 << "Description: Results from the BQRRP GPU runtime breakdown benchmark, recording the time it takes to perform every subroutine in BQRRP."
                "\nFile format: 15 data columns, each corresponding to a given BQRRP subroutine: preallocation_t_dur, qrcp_main_t_dur, copy_A_sk_t_dur, qrcp_piv_t_dur, copy_A_t_dur, piv_A_t_dur, copy_J_t_dur, updating_J_t_dur, preconditioning_t_dur, qr_tall_t_dur, q_reconstruction_t_dur, apply_transq_t_dur, sample_update_t_dur, t_rest, total_t_dur"
                "               rows correspond to BQRRP runs with block sizes varying in a way unique for a particular run."
                "\nInput type:"       + std::to_string(m_info.m_type) +
                "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
                "\nAdditional parameters: Tall QR subroutine geqrf BQRRP block sizes: " + b_sz_string +
                "\n";
        file2.flush();
    }

    std::string* file_name_3 = new std::string("_BQRRP_GPU_speed_comparisons_block_size_num_info_lines_" + std::to_string(6) + ".txt");
    std::ofstream file3(*file_name_3, std::ios::out | std::ios::app);

    file3 << "Description: Results from the BQRRP GPU speed comparison benchmark, recording the time it takes to perform BQRRP and alternative QR and QRCP factorizations."
            "\nFile format: 3 columns, containing time for each algorithm: BQRRP+CholQR, BQRRP+QRF, QRF;"
            "               rows correspond to BQRRP runs with block sizes varying in powers of 2 or multiples of 10"
            "\nInput type:"       + std::to_string(m_info.m_type) +
            "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
            "\nAdditional parameters: BQRRP block sizes: " + b_sz_string +
            "\n";
    file3.flush();

    auto start_time_all = steady_clock::now();
    for(size_t i = 0; i < b_sz.size(); ++i) {
        bench_BQRRP(profile_runtime, run_qrf, m_info, m, n, b_sz[i], all_data, state, file_name_1, file_name_2, file_name_3);
    }
    auto stop_time_all = steady_clock::now();
    long dur_time_all = duration_cast<microseconds>(stop_time_all - start_time_all).count();
    file3 << "Total benchmark execution time:" +  std::to_string(dur_time_all) + "\n";
    file3.flush();
}

static void run_mat_size_sweep(
    std::vector<int64_t> m_sz,
    int64_t b_sz,
    bool profile_runtime,
    bool run_qrf
){
    // Get a string representation of the matrix size vector
    std::string m_sz_string = std::accumulate(m_sz.begin(), m_sz.end(), std::string(),
                                [](const std::string& a, int b) {
                                    return a.empty() ? std::to_string(b) : a + "," + std::to_string(b);
                                });

    auto state = RandBLAS::RNGState();

    int64_t m_max = *std::max_element(m_sz.begin(), m_sz.end());
    BQRRPBenchData<double> all_data(m_max, m_max);
    RandLAPACK::gen::mat_gen_info<double> m_info(m_max, m_max, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
    cudaMemcpy(all_data.A_device, all_data.A.data(), m_max * m_max * sizeof(double), cudaMemcpyHostToDevice);

    std::string* file_name = new std::string("BQRRP_GPU_speed_comparisons_mat_size_num_info_lines_" + std::to_string(6) + ".txt");

    std::ofstream file(*file_name, std::ios::out | std::ios::app);
    file << "Description: Results from the BQRRP GPU speed comparison benchmark, recording the time it takes to perform BQRRP and alternative QR and QRCP factorizations."
            "\nFile format: 3 columns, containing time for each algorithm: BQRRP+CholQR, BQRRP+QRF, QRF;"
            "               rows correspond to BQRRP runs with varying mat sizes, with numruns repititions of each mat size."
            "\nInput type:"       + std::to_string(m_info.m_type) +
            "\nInput size:"       + " dim start: " + m_sz_string +
            "\nAdditional parameters: BQRRP block size: " + std::to_string(b_sz) +
            "\n";
    file.flush();

    for(size_t i = 0; i < m_sz.size(); ++i) {
        bench_BQRRP(profile_runtime, run_qrf, m_info, m_sz[i], m_sz[i], m_sz[i]/32, all_data, state, nullptr, nullptr, file_name);
    }
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <mode> [options]\n\n";
    std::cout << "Modes:\n";
    std::cout << "  block_size    Sweep over different block sizes with fixed matrix size\n";
    std::cout << "  mat_size      Sweep over different matrix sizes with fixed block size\n\n";
    std::cout << "Block size mode usage:\n";
    std::cout << "  " << prog_name << " block_size [matrix_size] [profile_runtime] [run_qrf]\n";
    std::cout << "    matrix_size:      Size of square matrix (e.g., 4096, 8192, 16384, 32768)\n";
    std::cout << "    profile_runtime:  1 to enable detailed profiling, 0 to disable (default: 0)\n";
    std::cout << "    run_qrf:          1 to run QRF comparison, 0 to skip (default: 1)\n\n";
    std::cout << "Matrix size mode usage:\n";
    std::cout << "  " << prog_name << " mat_size [profile_runtime] [run_qrf]\n";
    std::cout << "    profile_runtime:  1 to enable detailed profiling, 0 to disable (default: 0)\n";
    std::cout << "    run_qrf:          1 to run QRF comparison, 0 to skip (default: 1)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog_name << " block_size 16384\n";
    std::cout << "  " << prog_name << " block_size 32768 1 1\n";
    std::cout << "  " << prog_name << " mat_size\n";
    std::cout << "  " << prog_name << " mat_size 0 1\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "block_size") {
        // Default values
        int64_t matrix_size = 16384;
        bool profile_runtime = false;
        bool run_qrf = true;

        // Parse optional arguments
        if (argc > 2) matrix_size = std::stoll(argv[2]);
        if (argc > 3) profile_runtime = (std::stoi(argv[3]) != 0);
        if (argc > 4) run_qrf = (std::stoi(argv[4]) != 0);

        std::cout << "Running block size sweep benchmark\n";
        std::cout << "Matrix size: " << matrix_size << " x " << matrix_size << "\n";
        std::cout << "Profile runtime: " << (profile_runtime ? "yes" : "no") << "\n";
        std::cout << "Run QRF: " << (run_qrf ? "yes" : "no") << "\n\n";

        std::vector<int64_t> b_sz = {32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048};
        run_block_size_sweep(matrix_size, matrix_size, b_sz, profile_runtime, run_qrf);

    } else if (mode == "mat_size") {
        // Default values
        bool profile_runtime = false;
        bool run_qrf = true;

        // Parse optional arguments
        if (argc > 2) profile_runtime = (std::stoi(argv[2]) != 0);
        if (argc > 3) run_qrf = (std::stoi(argv[3]) != 0);

        std::cout << "Running matrix size sweep benchmark\n";
        std::cout << "Profile runtime: " << (profile_runtime ? "yes" : "no") << "\n";
        std::cout << "Run QRF: " << (run_qrf ? "yes" : "no") << "\n\n";

        std::vector<int64_t> m_sz = {512, 1024, 2048, 4096, 8192, 16384, 32768};
        int64_t b_sz = 0;  // Will be computed as m/32 in the benchmark
        run_mat_size_sweep(m_sz, b_sz, profile_runtime, run_qrf);

    } else {
        std::cerr << "Error: Unknown mode '" << mode << "'\n\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "\nBenchmark completed successfully!\n";
    return 0;
}
