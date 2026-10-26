#pragma once

// Scratch directory for tests that need real files.
//
// ctest runs every discovered test case in its own process, often in parallel,
// so a fixed or rand()-based path is shared between processes. Each ScratchDir
// is a fresh directory under std::filesystem::temp_directory_path(), named after
// the running test and the process id, and is removed with its contents when the
// object goes out of scope.

#include <gtest/gtest.h>

#include <atomic>
#include <cctype>
#include <filesystem>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace RandLAPACK_extras::testing {

class ScratchDir {
public:
    ScratchDir() {
        static std::atomic<unsigned> sequence{0};
        const std::string stem = "randlapack-" + test_label() + "-" +
                                 std::to_string(process_id()) + "-";
        // create_directory is atomic and returns false when the path already exists
        // as a directory, so one left by a crashed run or a reused pid only costs
        // another try.
        do {
            path_ = std::filesystem::temp_directory_path() / (stem + std::to_string(sequence++));
        } while (!std::filesystem::create_directory(path_));
    }

    ~ScratchDir() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    ScratchDir(const ScratchDir&) = delete;
    ScratchDir& operator=(const ScratchDir&) = delete;

    const std::filesystem::path& path() const { return path_; }

    /// Path of the file `name` inside this directory.
    std::string file(const std::string& name) const { return (path_ / name).string(); }

private:
    std::filesystem::path path_;

    // "Suite.test" of the running gtest case, reduced to filename-safe characters.
    static std::string test_label() {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string label = info ? std::string(info->test_suite_name()) + "." + info->name()
                                 : std::string("no-test");
        for (char& c : label)
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '.' && c != '_' && c != '-')
                c = '_';
        return label;
    }

    static long process_id() {
#ifdef _WIN32
        return static_cast<long>(_getpid());
#else
        return static_cast<long>(getpid());
#endif
    }
};

} // namespace RandLAPACK_extras::testing
