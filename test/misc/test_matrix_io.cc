#include "rl_matrix_io.hh"
#include "RandLAPACK.hh"
#include "rl_gen.hh"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

class TestMatrixIO : public ::testing::Test {
protected:
    std::filesystem::path directory;
    std::string filename;

    void SetUp() override {
        static std::atomic<unsigned> sequence{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        do {
            directory = std::filesystem::temp_directory_path() /
                ("randlapack-matrix-io-" + std::to_string(stamp) + "-" +
                 std::to_string(sequence++));
        } while (!std::filesystem::create_directory(directory));
        filename = (directory / "matrix").string();
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    void write_text(const std::string& contents) {
        std::ofstream file(filename, std::ios::binary);
        file << contents;
        ASSERT_TRUE(file.good());
    }

    void write_binary(int64_t m, int64_t n, const std::vector<double>& values) {
        std::ofstream file(filename, std::ios::binary);
        const int64_t dims[] = {m, n};
        file.write(reinterpret_cast<const char*>(dims), sizeof(dims));
        if (!values.empty())
            file.write(reinterpret_cast<const char*>(values.data()),
                       static_cast<std::streamsize>(values.size() * sizeof(double)));
        ASSERT_TRUE(file.good());
    }

    template <typename T>
    void check_text() {
        write_text(" 1\t-2.5 3e1\r\n4 5.25 +6");
        int64_t m = 99, n = 99;
        RandLAPACK::gen::read_txt_matrix<T>(m, n, nullptr, filename.c_str(), true);
        ASSERT_EQ(m, 2);
        ASSERT_EQ(n, 3);
        RandLAPACK::gen::read_txt_matrix<T>(m, n, nullptr, filename.c_str(), true);
        ASSERT_EQ(m, 2);
        ASSERT_EQ(n, 3);
        std::vector<T> values(6);
        RandLAPACK::gen::read_txt_matrix<T>(m, n, values.data(), filename.c_str(), false);
        EXPECT_EQ(values, (std::vector<T>{1, 4, -2.5, 5.25, 30, 6}));
    }

    template <typename T>
    void check_binary() {
        write_binary(2, 3, {1, -2.5, 30, 4, 5.25, 6});
        int64_t m = 0, n = 0;
        RandLAPACK::gen::read_bin_matrix<T>(m, n, nullptr, filename.c_str(), true);
        ASSERT_EQ(m, 2);
        ASSERT_EQ(n, 3);
        std::vector<T> values(6);
        RandLAPACK::gen::read_bin_matrix<T>(m, n, values.data(), filename.c_str(), false);
        EXPECT_EQ(values, (std::vector<T>{1, 4, -2.5, 5.25, 30, 6}));
    }
};

TEST_F(TestMatrixIO, TextFloatColumnMajor) { check_text<float>(); }
TEST_F(TestMatrixIO, TextDoubleColumnMajor) { check_text<double>(); }
TEST_F(TestMatrixIO, BinaryFloatColumnMajor) { check_binary<float>(); }
TEST_F(TestMatrixIO, BinaryDoubleColumnMajor) { check_binary<double>(); }

TEST_F(TestMatrixIO, RejectsMalformedText) {
    for (const auto& contents : {"", "\n", "1 2\n\n", "1 2\n3", "1 2\n3 4 5",
                                 "1 junk\n3 4", "1 2x\n3 4", "1 2\n3 1e9999"}) {
        SCOPED_TRACE(contents);
        write_text(contents);
        int64_t m = 2, n = 2;
        EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
            m, n, nullptr, filename.c_str(), true), std::runtime_error);
        EXPECT_EQ(m, 2);
        EXPECT_EQ(n, 2);
        std::vector<double> values(4);
        EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
            m, n, values.data(), filename.c_str(), false), std::runtime_error);
    }
}

TEST_F(TestMatrixIO, TextRejectsChangedShapeWithinBufferBounds) {
    write_text("1 2\n3 4\n");
    int64_t m = 0, n = 0;
    RandLAPACK::gen::read_txt_matrix<double>(m, n, nullptr, filename.c_str(), true);
    for (const auto& contents : {"1 2 3\n4 5 6\n", "1 2\n3 4\n5 6\n", "1 2\n"}) {
        write_text(contents);
        std::vector<double> values(6, -123);
        EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
            m, n, values.data() + 1, filename.c_str(), false), std::runtime_error);
        EXPECT_EQ(m, 2);
        EXPECT_EQ(n, 2);
        EXPECT_EQ(values.front(), -123);
        EXPECT_EQ(values.back(), -123);
    }
}

TEST_F(TestMatrixIO, RejectsEmbeddedControlBytes) {
    // Text-mode input on Windows can mistake Ctrl-Z for EOF and accept a prefix.
    for (char control : {'\0', '\x1a'}) {
        write_text(std::string("1 2\n") + control + "3 4\n");
        int64_t m = 2, n = 2;
        std::vector<double> values(4);
        EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
            m, n, nullptr, filename.c_str(), true), std::runtime_error);
        EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
            m, n, values.data(), filename.c_str(), false), std::runtime_error);
    }
}

TEST_F(TestMatrixIO, BinaryRejectsChangedShapeBeforeWriting) {
    write_binary(2, 2, {1, 2, 3, 4});
    int64_t m = 0, n = 0;
    RandLAPACK::gen::read_bin_matrix<double>(m, n, nullptr, filename.c_str(), true);
    write_binary(2, 3, {1, 2, 3, 4, 5, 6});
    std::vector<double> values(4, -123);
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        m, n, values.data(), filename.c_str(), false), std::runtime_error);
    EXPECT_EQ(m, 2);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(values, std::vector<double>(4, -123));
}

TEST_F(TestMatrixIO, RejectsTruncatedBinary) {
    for (const auto& contents : {"", "short header"}) {
        write_text(contents);
        int64_t m = 2, n = 2;
        EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
            m, n, nullptr, filename.c_str(), true), std::runtime_error);
    }
    write_binary(2, 2, {1, 2, 3});
    int64_t m = 2, n = 2;
    std::vector<double> values(4);
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        m, n, nullptr, filename.c_str(), true), std::runtime_error);
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        m, n, values.data(), filename.c_str(), false), std::runtime_error);
}

TEST_F(TestMatrixIO, RejectsExcessBinaryPayload) {
    write_binary(2, 2, {1, 2, 3, 4, 5});
    int64_t m = 2, n = 2;
    std::vector<double> values(4);
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        m, n, nullptr, filename.c_str(), true), std::runtime_error);
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        m, n, values.data(), filename.c_str(), false), std::runtime_error);
}

TEST_F(TestMatrixIO, RejectsInvalidBinaryDimensions) {
    const int64_t invalid[][2] = {{0, 2}, {2, 0}, {-1, 2}, {2, -1},
        {std::numeric_limits<int64_t>::max(), 2},
        {std::numeric_limits<int64_t>::max() / 4, 1}};
    for (const auto& dims : invalid) {
        SCOPED_TRACE(::testing::Message() << dims[0] << " x " << dims[1]);
        write_binary(dims[0], dims[1], {});
        int64_t m = 2, n = 2;
        EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
            m, n, nullptr, filename.c_str(), true), std::runtime_error);
        EXPECT_EQ(m, 2);
        EXPECT_EQ(n, 2);
    }
}

TEST_F(TestMatrixIO, RejectsInvalidOutputArguments) {
    write_text("1 2\n3 4\n");
    std::vector<double> values(4);
    for (int64_t invalid : {int64_t(0), int64_t(-1), std::numeric_limits<int64_t>::max()}) {
        int64_t m = invalid, n = 2;
        EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
            m, n, values.data(), filename.c_str(), false), std::runtime_error);
    }
    int64_t m = 2, n = 2;
    EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
        m, n, nullptr, filename.c_str(), false), std::runtime_error);
    write_binary(2, 2, {1, 2, 3, 4});
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        m, n, nullptr, filename.c_str(), false), std::runtime_error);
    m = 0;
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        m, n, values.data(), filename.c_str(), false), std::runtime_error);
    EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
        m, n, nullptr, nullptr, true), std::runtime_error);
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        m, n, nullptr, nullptr, true), std::runtime_error);
}

TEST_F(TestMatrixIO, RejectsMissingFiles) {
    int64_t m = 2, n = 2;
    std::vector<double> values(4);
    for (bool query : {false, true}) {
        EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
            m, n, values.data(), filename.c_str(), query), std::runtime_error);
        EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
            m, n, values.data(), filename.c_str(), query), std::runtime_error);
    }
}

TEST_F(TestMatrixIO, LegacyReaderPreservesQueryProtocol) {
    write_text("1 2 3\n4 5 6\n");
    int64_t m = 0, n = 0;
    int query = 1;
    RandLAPACK::gen::process_input_mat<double>(m, n, nullptr, filename.data(), query);
    EXPECT_EQ(query, 0);
    ASSERT_EQ(m, 2);
    ASSERT_EQ(n, 3);
    query = 1;
    RandLAPACK::gen::process_input_mat<double>(m, n, nullptr, filename.data(), query);
    ASSERT_EQ(m, 2);
    ASSERT_EQ(n, 3);
    std::vector<double> values(6);
    RandLAPACK::gen::process_input_mat(m, n, values.data(), filename.data(), query);
    EXPECT_EQ(values, (std::vector<double>{1, 4, 2, 5, 3, 6}));
    write_text("1 2 3\n4 broken 6\n");
    query = 1;
    EXPECT_THROW(RandLAPACK::gen::process_input_mat<double>(
        m, n, nullptr, filename.data(), query), std::runtime_error);
    EXPECT_EQ(query, 1);
}

TEST_F(TestMatrixIO, CustomInputGeneratorUsesReader) {
    write_text("1 2 3\n4 5 6\n");
    int64_t m = 0, n = 0;
    RandLAPACK::gen::mat_gen_info<double> info(m, n, RandLAPACK::gen::custom_input);
    info.filename = filename.data();
    info.workspace_query_mod = 1;
    RandBLAS::RNGState<> state(42);
    const auto counter_before = state.counter;
    RandLAPACK::gen::mat_gen<double>(info, nullptr, state);
    EXPECT_EQ(info.workspace_query_mod, 0);
    ASSERT_EQ(info.rows, 2);
    ASSERT_EQ(info.cols, 3);
    std::vector<double> values(6);
    RandLAPACK::gen::mat_gen(info, values.data(), state);
    EXPECT_EQ(values, (std::vector<double>{1, 4, 2, 5, 3, 6}));
    EXPECT_EQ(state.counter, counter_before);
    write_text("1 2 3\n4 broken 6\n");
    EXPECT_THROW(RandLAPACK::gen::mat_gen(info, values.data(), state), std::runtime_error);
}

} // namespace
