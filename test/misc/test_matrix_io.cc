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
    void read_matrix(bool binary, Layout layout, int64_t& m, int64_t& n,
                     T* values, int64_t lda, bool query) {
        if (binary)
            RandLAPACK::gen::read_bin_matrix<T>(
                layout, m, n, values, lda, filename.c_str(), query);
        else
            RandLAPACK::gen::read_txt_matrix<T>(
                layout, m, n, values, lda, filename.c_str(), query);
    }

    template <typename T>
    void check_layouts(bool binary) {
        if (binary)
            write_binary(2, 3, {1, -2.5, 30, 4, 5.25, 6});
        else
            write_text(" 1\t-2.5 3e1\r\n4 5.25 +6");
        for (Layout layout : {Layout::ColMajor, Layout::RowMajor}) {
            for (bool padded : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "column-major=" <<
                             (layout == Layout::ColMajor) << ", padded=" << padded);
                int64_t m = 99, n = 99;
                read_matrix<T>(binary, layout, m, n, nullptr, 0, true);
                ASSERT_EQ(m, 2);
                ASSERT_EQ(n, 3);
                const int64_t lda = (layout == Layout::ColMajor ? m : n) +
                                    (padded ? 2 : 0);
                const std::vector<T> expected = layout == Layout::ColMajor ?
                    (padded ? std::vector<T>{-123, 1, 4, -123, -123,
                        -2.5, 5.25, -123, -123, 30, 6, -123, -123, -123} :
                        std::vector<T>{-123, 1, 4, -2.5, 5.25, 30, 6, -123}) :
                    (padded ? std::vector<T>{-123, 1, -2.5, 30, -123, -123,
                        4, 5.25, 6, -123, -123, -123} :
                        std::vector<T>{-123, 1, -2.5, 30, 4, 5.25, 6, -123});
                std::vector<T> values(expected.size(), -123);
                read_matrix(binary, layout, m, n, values.data() + 1, lda, false);
                EXPECT_EQ(values, expected);
            }
        }
    }
};

TEST_F(TestMatrixIO, TextFloatLayoutsAndStrides) { check_layouts<float>(false); }
TEST_F(TestMatrixIO, TextDoubleLayoutsAndStrides) { check_layouts<double>(false); }
TEST_F(TestMatrixIO, BinaryFloatLayoutsAndStrides) { check_layouts<float>(true); }
TEST_F(TestMatrixIO, BinaryDoubleLayoutsAndStrides) { check_layouts<double>(true); }

TEST_F(TestMatrixIO, AllowsUnusedLargeStride) {
    for (bool binary : {false, true}) {
        for (Layout layout : {Layout::ColMajor, Layout::RowMajor}) {
            int64_t m = layout == Layout::ColMajor ? 3 : 1;
            int64_t n = layout == Layout::ColMajor ? 1 : 3;
            if (binary)
                write_binary(m, n, {1, 2, 3});
            else
                write_text(layout == Layout::ColMajor ? "1\n2\n3\n" : "1 2 3\n");
            std::vector<double> values(5, -123);
            read_matrix(binary, layout, m, n, values.data() + 1,
                        std::numeric_limits<int64_t>::max(), false);
            EXPECT_EQ(values, (std::vector<double>{-123, 1, 2, 3, -123}));
        }
    }
}

TEST_F(TestMatrixIO, QueriesIgnoreOutputBufferAndStride) {
    for (bool binary : {false, true}) {
        if (binary)
            write_binary(2, 3, {1, 2, 3, 4, 5, 6});
        else
            write_text("1 2 3\n4 5 6\n");
        for (Layout layout : {Layout::ColMajor, Layout::RowMajor}) {
            for (int64_t lda : {int64_t(-1), int64_t(0),
                                std::numeric_limits<int64_t>::max()}) {
                int64_t m = -1, n = -1;
                read_matrix<double>(binary, layout, m, n, nullptr, lda, true);
                EXPECT_EQ(m, 2);
                EXPECT_EQ(n, 3);
                read_matrix<double>(binary, layout, m, n, nullptr, lda, true);
                EXPECT_EQ(m, 2);
                EXPECT_EQ(n, 3);
            }
        }
    }
}

TEST_F(TestMatrixIO, RejectsInvalidLayoutBeforeWriting) {
    for (bool binary : {false, true}) {
        if (binary)
            write_binary(2, 3, {1, 2, 3, 4, 5, 6});
        else
            write_text("1 2 3\n4 5 6\n");
        for (bool query : {false, true}) {
            int64_t m = 2, n = 3;
            std::vector<double> values(6, -123);
            EXPECT_THROW(read_matrix(binary, static_cast<Layout>(0), m, n,
                         values.data(), 3, query), RandLAPACK::Error);
            EXPECT_EQ(m, 2);
            EXPECT_EQ(n, 3);
            EXPECT_EQ(values, std::vector<double>(6, -123));
        }
    }
}

TEST_F(TestMatrixIO, RejectsInvalidStrideBeforeWriting) {
    for (bool binary : {false, true}) {
        if (binary)
            write_binary(2, 3, {1, 2, 3, 4, 5, 6});
        else
            write_text("1 2 3\n4 5 6\n");
        for (Layout layout : {Layout::ColMajor, Layout::RowMajor}) {
            const int64_t minimum = layout == Layout::ColMajor ? 2 : 3;
            for (int64_t lda : {int64_t(-1), int64_t(0), minimum - 1,
                                std::numeric_limits<int64_t>::max(),
                                std::numeric_limits<int64_t>::max() /
                                    static_cast<int64_t>(sizeof(double))}) {
                SCOPED_TRACE(::testing::Message() << "binary=" << binary <<
                             ", column-major=" << (layout == Layout::ColMajor) <<
                             ", lda=" << lda);
                int64_t m = 2, n = 3;
                std::vector<double> values(6, -123);
                EXPECT_THROW(read_matrix(binary, layout, m, n, values.data(),
                             lda, false), RandLAPACK::Error);
                EXPECT_EQ(values, std::vector<double>(6, -123));
            }
        }
    }
}

TEST_F(TestMatrixIO, RejectsMalformedText) {
    for (const auto& contents : {"", "\n", "1 2\n\n", "1 2\n3", "1 2\n3 4 5",
                                 "1 junk\n3 4", "1 2x\n3 4", "1 2\n3 1e9999"}) {
        SCOPED_TRACE(contents);
        write_text(contents);
        int64_t m = 2, n = 2;
        EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
            Layout::ColMajor, m, n, nullptr, 0, filename.c_str(), true), RandLAPACK::Error);
        EXPECT_EQ(m, 2);
        EXPECT_EQ(n, 2);
        std::vector<double> values(4);
        EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
            Layout::ColMajor, m, n, values.data(), m, filename.c_str(), false), RandLAPACK::Error);
    }
}

TEST_F(TestMatrixIO, TextRejectsChangedShapeWithinBufferBounds) {
    write_text("1 2\n3 4\n");
    int64_t m = 0, n = 0;
    RandLAPACK::gen::read_txt_matrix<double>(
        Layout::ColMajor, m, n, nullptr, 0, filename.c_str(), true);
    for (const auto& contents : {"1 2 3\n4 5 6\n", "1 2\n3 4\n5 6\n", "1 2\n"}) {
        write_text(contents);
        std::vector<double> values(6, -123);
        EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
            Layout::ColMajor, m, n, values.data() + 1, m, filename.c_str(), false), RandLAPACK::Error);
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
            Layout::ColMajor, m, n, nullptr, 0, filename.c_str(), true), RandLAPACK::Error);
        EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
            Layout::ColMajor, m, n, values.data(), m, filename.c_str(), false), RandLAPACK::Error);
    }
}

TEST_F(TestMatrixIO, BinaryRejectsChangedShapeBeforeWriting) {
    write_binary(2, 2, {1, 2, 3, 4});
    int64_t m = 0, n = 0;
    RandLAPACK::gen::read_bin_matrix<double>(
        Layout::ColMajor, m, n, nullptr, 0, filename.c_str(), true);
    write_binary(2, 3, {1, 2, 3, 4, 5, 6});
    std::vector<double> values(4, -123);
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        Layout::ColMajor, m, n, values.data(), m, filename.c_str(), false), RandLAPACK::Error);
    EXPECT_EQ(m, 2);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(values, std::vector<double>(4, -123));
}

TEST_F(TestMatrixIO, RejectsTruncatedBinary) {
    for (const auto& contents : {"", "short header"}) {
        write_text(contents);
        int64_t m = 2, n = 2;
        EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
            Layout::ColMajor, m, n, nullptr, 0, filename.c_str(), true), RandLAPACK::Error);
    }
    write_binary(2, 2, {1, 2, 3});
    int64_t m = 2, n = 2;
    std::vector<double> values(4);
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        Layout::ColMajor, m, n, nullptr, 0, filename.c_str(), true), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        Layout::ColMajor, m, n, values.data(), m, filename.c_str(), false), RandLAPACK::Error);
}

TEST_F(TestMatrixIO, RejectsExcessBinaryPayload) {
    write_binary(2, 2, {1, 2, 3, 4, 5});
    int64_t m = 2, n = 2;
    std::vector<double> values(4);
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        Layout::ColMajor, m, n, nullptr, 0, filename.c_str(), true), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        Layout::ColMajor, m, n, values.data(), m, filename.c_str(), false), RandLAPACK::Error);
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
            Layout::ColMajor, m, n, nullptr, 0, filename.c_str(), true), RandLAPACK::Error);
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
            Layout::ColMajor, m, n, values.data(), m, filename.c_str(), false), RandLAPACK::Error);
    }
    int64_t m = 2, n = 2;
    EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
        Layout::ColMajor, m, n, nullptr, m, filename.c_str(), false), RandLAPACK::Error);
    write_binary(2, 2, {1, 2, 3, 4});
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        Layout::ColMajor, m, n, nullptr, m, filename.c_str(), false), RandLAPACK::Error);
    m = 0;
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        Layout::ColMajor, m, n, values.data(), m, filename.c_str(), false), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
        Layout::ColMajor, m, n, nullptr, 0, nullptr, true), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
        Layout::ColMajor, m, n, nullptr, 0, nullptr, true), RandLAPACK::Error);
}

TEST_F(TestMatrixIO, RejectsMissingFiles) {
    int64_t m = 2, n = 2;
    std::vector<double> values(4);
    for (bool query : {false, true}) {
        EXPECT_THROW(RandLAPACK::gen::read_txt_matrix<double>(
            Layout::ColMajor, m, n, values.data(), m, filename.c_str(), query), RandLAPACK::Error);
        EXPECT_THROW(RandLAPACK::gen::read_bin_matrix<double>(
            Layout::ColMajor, m, n, values.data(), m, filename.c_str(), query), RandLAPACK::Error);
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
        m, n, nullptr, filename.data(), query), RandLAPACK::Error);
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
    EXPECT_THROW(RandLAPACK::gen::mat_gen(info, values.data(), state), RandLAPACK::Error);
}

} // namespace
