#include "../../misc/ext_matrix_io.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

class MatrixLoaderFiles : public ::testing::Test {
protected:
    std::filesystem::path directory;

    void SetUp() override {
        static std::atomic<unsigned> sequence{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        do {
            directory = std::filesystem::temp_directory_path() /
                ("randlapack-matrix-loader-" + std::to_string(stamp) + "-" +
                 std::to_string(sequence++));
        } while (!std::filesystem::create_directory(directory));
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    std::string path(const std::string& name) const {
        return (directory / name).string();
    }

    void write_text(const std::string& name, const std::string& contents) {
        std::ofstream file(path(name), std::ios::binary);
        file << contents;
        file.close();
        ASSERT_TRUE(file.good());
    }

    void write_binary(const std::string& name, int64_t m, int64_t n,
                      const std::vector<double>& values) {
        std::ofstream file(path(name), std::ios::binary);
        const int64_t dims[] = {m, n};
        file.write(reinterpret_cast<const char*>(dims), sizeof(dims));
        if (!values.empty())
            file.write(reinterpret_cast<const char*>(values.data()),
                       static_cast<std::streamsize>(values.size() * sizeof(double)));
        file.close();
        ASSERT_TRUE(file.good());
    }

    void write_dense(const std::string& name, const std::string& format) {
        if (format == "binary") {
            write_binary(name, 4, 6, {1, -2.5, 30, 4, 5.25, 6,
                                     7, 8, 9, 10, 11, 12,
                                     13, 14, 15, 16, 17, 18,
                                     19, 20, 21, 22, 23, 24});
        } else if (format == "array") {
            write_text(name, "%%MatrixMarket matrix array real general\n4 6\n"
                       "1\n7\n13\n19\n-2.5\n8\n14\n20\n30\n9\n15\n21\n"
                       "4\n10\n16\n22\n5.25\n11\n17\n23\n6\n12\n18\n24\n");
        } else {
            write_text(name, "1 -2.5 30 4 5.25 6\n7 8 9 10 11 12\n"
                       "13 14 15 16 17 18\n19 20 21 22 23 24\n");
        }
    }
};

template <typename T>
class TestExtMatrixIO : public MatrixLoaderFiles {
protected:
    void check_dense(BenchIO::LoadedMatrix<T>& matrix, int64_t m, int64_t n,
                     const std::vector<T>& expected) {
        EXPECT_EQ(matrix.m, m);
        EXPECT_EQ(matrix.n, n);
        EXPECT_FALSE(matrix.is_sparse);
        EXPECT_FALSE(matrix.csc);
        EXPECT_FALSE(matrix.eigen_sparse);
        EXPECT_EQ(matrix.dense_data, expected);
        EXPECT_EQ(matrix.data(), matrix.dense_data.data());
    }

    void check_full(const std::string& name, const std::string& format) {
        this->write_dense(name, format);
        auto matrix = BenchIO::load_matrix<T>(this->path(name));
        check_dense(matrix, 4, 6, {1, 7, 13, 19, -2.5, 8, 14, 20,
                                 30, 9, 15, 21, 4, 10, 16, 22,
                                 5.25, 11, 17, 23, 6, 12, 18, 24});
    }

    void check_crop(const std::string& name, const std::string& format) {
        this->write_dense(name, format);
        auto matrix = BenchIO::load_matrix<T>(this->path(name), 0.5);
        check_dense(matrix, 2, 3, {1, 7, -2.5, 8, 30, 9});
        auto smaller = BenchIO::load_matrix<T>(this->path(name), 0.49);
        check_dense(smaller, 1, 2, {1, -2.5});
    }

    void check_sparse(const BenchIO::LoadedMatrix<T>& matrix, int64_t m, int64_t n,
                      const std::vector<T>& expected) {
        ASSERT_EQ(matrix.m, m);
        ASSERT_EQ(matrix.n, n);
        ASSERT_TRUE(matrix.is_sparse);
        EXPECT_TRUE(matrix.dense_data.empty());
        ASSERT_TRUE(matrix.csc);
        ASSERT_TRUE(matrix.eigen_sparse);
        ASSERT_EQ(matrix.csc->n_rows, m);
        ASSERT_EQ(matrix.csc->n_cols, n);
        ASSERT_EQ(matrix.eigen_sparse->rows(), m);
        ASSERT_EQ(matrix.eigen_sparse->cols(), n);
        ASSERT_NE(matrix.csc->colptr, nullptr);
        ASSERT_EQ(expected.size(), static_cast<size_t>(m * n));

        ASSERT_EQ(matrix.csc->colptr[0], 0);
        ASSERT_EQ(matrix.csc->colptr[n], matrix.csc->nnz);
        for (int64_t col = 0; col < n; ++col) {
            ASSERT_LE(matrix.csc->colptr[col], matrix.csc->colptr[col + 1]);
            for (int64_t k = matrix.csc->colptr[col]; k < matrix.csc->colptr[col + 1]; ++k) {
                const auto row = matrix.csc->rowidxs[k];
                ASSERT_GE(row, 0);
                ASSERT_LT(row, m);
            }
            for (int64_t row = 0; row < m; ++row)
                EXPECT_EQ(matrix.eigen_sparse->coeff(row, col), expected[row + m * col]);
        }
        // Public conversion must agree with Eigen, including duplicate inputs.
        std::vector<T> csc_dense(m * n, -123);
        RandBLAS::sparse_data::csc::csc_to_dense(
            *matrix.csc, blas::Layout::ColMajor, csc_dense.data());
        EXPECT_EQ(csc_dense, expected);

        // Exercise the storage through both libraries, including zero-nnz input.
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> rhs(n, 2);
        for (int64_t row = 0; row < n; ++row) {
            rhs(row, 0) = static_cast<T>(row + 1);
            rhs(row, 1) = static_cast<T>(2 - row);
        }
        std::vector<T> product(m * 2, -123);
        RandBLAS::sparse_data::left_spmm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            m, 2, n, T(1), *matrix.csc, 0, 0, rhs.data(), n,
            T(0), product.data(), m);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_product =
            *matrix.eigen_sparse * rhs;
        for (int64_t col = 0; col < 2; ++col) {
            for (int64_t row = 0; row < m; ++row) {
                T reference = 0;
                for (int64_t k = 0; k < n; ++k)
                    reference += expected[row + m * k] * rhs(k, col);
                EXPECT_EQ(product[row + m * col], reference);
                EXPECT_EQ(eigen_product(row, col), reference);
            }
        }
    }
};

using MatrixLoaderScalars = ::testing::Types<float, double>;
TYPED_TEST_SUITE(TestExtMatrixIO, MatrixLoaderScalars);

TYPED_TEST(TestExtMatrixIO, TextColumnMajor) { this->check_full("matrix.txt", "text"); }
TYPED_TEST(TestExtMatrixIO, BinaryColumnMajor) { this->check_full("matrix.bin", "binary"); }
TYPED_TEST(TestExtMatrixIO, ArrayColumnMajor) { this->check_full("matrix.mtx", "array"); }
TYPED_TEST(TestExtMatrixIO, TextCrop) { this->check_crop("matrix.txt", "text"); }
TYPED_TEST(TestExtMatrixIO, BinaryCrop) { this->check_crop("matrix.bin", "binary"); }
TYPED_TEST(TestExtMatrixIO, ArrayCrop) { this->check_crop("matrix.mtx", "array"); }

TYPED_TEST(TestExtMatrixIO, Extensionless) {
    this->check_full("matrix", "text");
}

TYPED_TEST(TestExtMatrixIO, CaseInsensitiveAndUnknownExtensions) {
    this->check_full("matrix.BiN", "binary");
    this->check_full("matrix.MTX", "array");
    this->check_full("matrix.TXT", "text");
    this->check_full("matrix.dat", "text");
}

TYPED_TEST(TestExtMatrixIO, CoordinateAndCrop) {
    this->write_text("coordinate.mtx", "%%MatrixMarket matrix coordinate real general\n"
                     "% Deliberately unordered coordinates.\n4 6 5\n"
                     "4 6 9\n2 3 -2.5\n1 1 1\n3 2 7\n1 3 3\n");
    auto full = BenchIO::load_matrix<TypeParam>(this->path("coordinate.mtx"));
    this->check_sparse(full, 4, 6, {1, 0, 0, 0, 0, 0, 7, 0,
                                  3, -2.5, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 9});
    auto crop = BenchIO::load_matrix<TypeParam>(this->path("coordinate.mtx"), 0.5);
    this->check_sparse(crop, 2, 3, {1, 0, 0, 0, 3, -2.5});
}

TYPED_TEST(TestExtMatrixIO, SymmetricCoordinatesAreExpanded) {
    this->write_text("symmetric.mtx", "%%MatrixMarket matrix coordinate real symmetric\n"
                     "3 3 4\n1 1 2\n2 1 -3\n3 2 4\n3 3 5\n");
    auto matrix = BenchIO::load_matrix<TypeParam>(this->path("symmetric.mtx"));
    this->check_sparse(matrix, 3, 3, {2, -3, 0, -3, 0, 4, 0, 4, 5});
    auto crop = BenchIO::load_matrix<TypeParam>(this->path("symmetric.mtx"), 0.75);
    this->check_sparse(crop, 2, 2, {2, -3, -3, 0});
}

TYPED_TEST(TestExtMatrixIO, PatternCoordinatesHaveUnitValues) {
    this->write_text("pattern.mtx", "%%MatrixMarket matrix coordinate pattern general\n"
                     "3 4 3\n3 4\n1 2\n2 1\n");
    auto matrix = BenchIO::load_matrix<TypeParam>(this->path("pattern.mtx"));
    this->check_sparse(matrix, 3, 4, {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1});
}

TYPED_TEST(TestExtMatrixIO, DuplicateCoordinatesAreSummed) {
    this->write_text("duplicates.mtx", "%%MatrixMarket matrix coordinate real general\n"
                     "2 3 5\n1 2 2\n2 3 4\n1 2 3\n2 1 -1\n2 3 -4\n");
    auto matrix = BenchIO::load_matrix<TypeParam>(this->path("duplicates.mtx"));
    this->check_sparse(matrix, 2, 3, {0, -1, 5, 0, 0, 0});
}

TYPED_TEST(TestExtMatrixIO, EmptySparseMatrixCanMultiply) {
    this->write_text("empty.mtx", "%%MatrixMarket matrix coordinate real general\n4 6 0\n");
    auto matrix = BenchIO::load_matrix<TypeParam>(this->path("empty.mtx"));
    this->check_sparse(matrix, 4, 6, std::vector<TypeParam>(24, 0));
    ASSERT_TRUE(matrix.csc);
    EXPECT_EQ(matrix.csc->nnz, 0);
}

TYPED_TEST(TestExtMatrixIO, SparseCropWithNoEntriesCanMultiply) {
    this->write_text("empty-crop.mtx", "%%MatrixMarket matrix coordinate real general\n"
                     "4 6 2\n4 1 2\n1 6 3\n");
    auto matrix = BenchIO::load_matrix<TypeParam>(this->path("empty-crop.mtx"), 0.5);
    this->check_sparse(matrix, 2, 3, std::vector<TypeParam>(6, 0));
    ASSERT_TRUE(matrix.csc);
    EXPECT_EQ(matrix.csc->nnz, 0);
}

TYPED_TEST(TestExtMatrixIO, SparseOwnershipSurvivesMoves) {
    using Matrix = BenchIO::LoadedMatrix<TypeParam>;
    EXPECT_FALSE(std::is_copy_constructible_v<Matrix>);
    EXPECT_FALSE(std::is_copy_assignable_v<Matrix>);
    EXPECT_TRUE(std::is_nothrow_move_constructible_v<Matrix>);
    EXPECT_TRUE(std::is_nothrow_move_assignable_v<Matrix>);
    if constexpr (std::is_copy_constructible_v<Matrix>) {
        GTEST_SKIP() << "Cannot exercise ownership transfer while the matrix is copyable";
    } else {
        this->write_text("move.mtx", "%%MatrixMarket matrix coordinate real general\n"
                         "2 3 2\n1 2 5\n2 3 -2\n");
        auto target = BenchIO::load_matrix<TypeParam>(this->path("move.mtx"));
        {
            auto source = BenchIO::load_matrix<TypeParam>(this->path("move.mtx"));
            auto* original_csc = &*source.csc;
            auto* original_eigen = &*source.eigen_sparse;
            Matrix moved(std::move(source));
            EXPECT_FALSE(source.csc);
            EXPECT_FALSE(source.eigen_sparse);
            EXPECT_EQ(&*moved.csc, original_csc);
            EXPECT_EQ(&*moved.eigen_sparse, original_eigen);
            this->check_sparse(moved, 2, 3, {0, 0, 5, 0, 0, -2});
            target = std::move(moved);
            EXPECT_FALSE(moved.csc);
            EXPECT_FALSE(moved.eigen_sparse);
            EXPECT_EQ(&*target.csc, original_csc);
            EXPECT_EQ(&*target.eigen_sparse, original_eigen);
        }
        this->check_sparse(target, 2, 3, {0, 0, 5, 0, 0, -2});
        // Replacing sparse data by dense data releases both sparse resources.
        this->write_dense("move.txt", "text");
        target = BenchIO::load_matrix<TypeParam>(this->path("move.txt"), 0.5);
        this->check_dense(target, 2, 3, {1, 7, -2.5, 8, 30, 9});
        Matrix dense(std::move(target));
        this->check_dense(dense, 2, 3, {1, 7, -2.5, 8, 30, 9});
    }
}

TYPED_TEST(TestExtMatrixIO, RejectsInvalidRatiosAndEmptyCrops) {
    this->write_dense("matrix.txt", "text");
    this->write_dense("matrix.bin", "binary");
    this->write_dense("matrix.mtx", "array");
    this->write_text("sparse.mtx", "%%MatrixMarket matrix coordinate real general\n"
                     "4 6 1\n1 1 2\n");
    for (const auto* name : {"matrix.txt", "matrix.bin", "matrix.mtx", "sparse.mtx"}) {
        for (double ratio : {-1.0, 0.0, 1.01, std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN(), 0.1}) {
            SCOPED_TRACE(::testing::Message() << name << ", ratio=" << ratio);
            EXPECT_THROW(BenchIO::load_matrix<TypeParam>(this->path(name), ratio),
                         RandLAPACK::Error);
        }
    }
}

TYPED_TEST(TestExtMatrixIO, MissingFilesThrowLoaderError) {
    for (const auto* name : {"missing.txt", "missing.bin", "missing.mtx", "missing"}) {
        SCOPED_TRACE(name);
        EXPECT_THROW(BenchIO::load_matrix<TypeParam>(this->path(name)), RandLAPACK::Error);
    }
}

TYPED_TEST(TestExtMatrixIO, MalformedDenseFilesThrowLoaderError) {
    for (const auto* contents : {"", "1 2\n3\n", "1 nope\n3 4\n"}) {
        this->write_text("invalid.txt", contents);
        EXPECT_THROW(BenchIO::load_matrix<TypeParam>(this->path("invalid.txt")),
                     RandLAPACK::Error);
    }
    this->write_text("invalid.bin", "short");
    EXPECT_THROW(BenchIO::load_matrix<TypeParam>(this->path("invalid.bin")),
                 RandLAPACK::Error);
    this->write_binary("invalid.bin", 2, 3, {1, 2});
    EXPECT_THROW(BenchIO::load_matrix<TypeParam>(this->path("invalid.bin")),
                 RandLAPACK::Error);
    this->write_binary("invalid.bin", 1, 1, {1, 2});
    EXPECT_THROW(BenchIO::load_matrix<TypeParam>(this->path("invalid.bin")),
                 RandLAPACK::Error);
}

TYPED_TEST(TestExtMatrixIO, MalformedMatrixMarketThrowsLoaderError) {
    for (const auto* contents : {
            "not a Matrix Market header\n",
            "%%MatrixMarket matrix array real general\n2 3\n1\n2\n",
            "%%MatrixMarket matrix array real general\n0 3\n",
            "%%MatrixMarket matrix coordinate real general\n3 3 1\n4 1 2\n",
            "%%MatrixMarket matrix coordinate real general\n3 3 1\n0 1 2\n",
            "%%MatrixMarket matrix coordinate real general\n3 3 2\n1 1 2\n",
            "%%MatrixMarket matrix coordinate real general\n3 3 1\n1 1 nope\n"}) {
        SCOPED_TRACE(contents);
        this->write_text("invalid.mtx", contents);
        EXPECT_THROW(BenchIO::load_matrix<TypeParam>(this->path("invalid.mtx")),
                     RandLAPACK::Error);
    }
}

TYPED_TEST(TestExtMatrixIO, RejectsUnrepresentableDimensionsBeforeAllocation) {
    const auto large = std::to_string(std::numeric_limits<int64_t>::max());
    // Two columns overflow signed indexing; one overflows the byte count.
    for (const auto* columns : {"1", "2"}) {
        this->write_text("huge.mtx", "%%MatrixMarket matrix array real general\n" +
                         large + " " + columns + "\n");
        EXPECT_THROW(BenchIO::load_matrix<TypeParam>(this->path("huge.mtx")),
                     RandLAPACK::Error);
    }
    this->write_binary("huge.bin", std::numeric_limits<int64_t>::max(), 2, {});
    EXPECT_THROW(BenchIO::load_matrix<TypeParam>(this->path("huge.bin")), RandLAPACK::Error);

    const auto large_index = std::to_string(static_cast<int64_t>(std::numeric_limits<int>::max()) + 1);
    for (const auto& dimensions : {large_index + " 1 0\n", "1 " + large_index + " 0\n",
                                   "1 1 " + large_index + "\n"}) {
        SCOPED_TRACE(dimensions);
        this->write_text("huge.mtx", "%%MatrixMarket matrix coordinate real general\n" + dimensions);
        EXPECT_THROW(BenchIO::load_matrix<TypeParam>(this->path("huge.mtx")),
                     RandLAPACK::Error);
    }
}

TYPED_TEST(TestExtMatrixIO, SymmetricArray) {
    this->write_text("symmetric-array.mtx", "%%MatrixMarket matrix array real symmetric\n"
                     "3 3\n1\n2\n3\n4\n5\n6\n");
    auto matrix = BenchIO::load_matrix<TypeParam>(this->path("symmetric-array.mtx"));
    this->check_dense(matrix, 3, 3, {1, 2, 3, 2, 4, 5, 3, 5, 6});
    auto crop = BenchIO::load_matrix<TypeParam>(this->path("symmetric-array.mtx"), 0.75);
    this->check_dense(crop, 2, 2, {1, 2, 2, 4});
}

TYPED_TEST(TestExtMatrixIO, SkewArray) {
    this->write_text("skew-array.mtx", "%%MatrixMarket matrix array real skew-symmetric\n"
                     "3 3\n2\n3\n5\n");
    auto matrix = BenchIO::load_matrix<TypeParam>(this->path("skew-array.mtx"));
    this->check_dense(matrix, 3, 3, {0, 2, 3, -2, 0, 5, -3, -5, 0});
    this->write_text("skew-scalar.mtx", "%%MatrixMarket matrix array real skew-symmetric\n1 1\n");
    auto scalar = BenchIO::load_matrix<TypeParam>(this->path("skew-scalar.mtx"));
    this->check_dense(scalar, 1, 1, {0});
}

TYPED_TEST(TestExtMatrixIO, TruncatedSymmetricArray) {
    for (const auto* symmetry : {"symmetric", "hermitian"}) {
        SCOPED_TRACE(symmetry);
        this->write_text("truncated-array.mtx",
                         std::string("%%MatrixMarket matrix array real ") + symmetry +
                         "\n3 3\n1\n2\n3\n4\n5\n");
        EXPECT_THROW(BenchIO::load_matrix<TypeParam>(this->path("truncated-array.mtx")),
                     RandLAPACK::Error);
    }
}

TYPED_TEST(TestExtMatrixIO, InvalidSkewArray) {
    for (const auto* payload : {"3 3\n2\n3\n", "3 3\n2\n3\n5\n7\n", "1 1\n7\n"}) {
        SCOPED_TRACE(payload);
        this->write_text("invalid-skew.mtx",
                         std::string("%%MatrixMarket matrix array real skew-symmetric\n") + payload);
        EXPECT_THROW(BenchIO::load_matrix<TypeParam>(this->path("invalid-skew.mtx")),
                     RandLAPACK::Error);
    }
}

} // namespace
