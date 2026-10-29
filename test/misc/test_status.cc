// The test suite defines RandLAPACK_ENABLE_TEST_HOOKS, so RL_TEST_FAULT consults test_hooks::fault.

#include "rl_status.hh"

#include <cstring>
#include <iterator>
#include <set>
#include <string>
#include <gtest/gtest.h>

using RandLAPACK::Status;

TEST(TestStatus, every_status_has_its_own_name) {
    const Status all[] = {
        Status::Success, Status::CholeskyFailure, Status::OrthogonalizationFailure,
        Status::NotConverged, Status::Breakdown, Status::NotPositiveSemidefinite,
        Status::NonFiniteInput, Status::NonFiniteResult,
    };
    std::set<std::string> names;
    for (Status s : all)
        names.insert(RandLAPACK::to_string(s));
    EXPECT_EQ(names.size(), std::size(all));
    EXPECT_EQ(names.count("unknown Status"), 0u);
    EXPECT_EQ(static_cast<int>(Status::Success), 0);
}

TEST(TestStatus, fault_hook) {
    EXPECT_FALSE(RL_TEST_FAULT("anything"));
    RandLAPACK::test_hooks::fault = [](const char* name) { return std::strcmp(name, "chosen") == 0; };
    EXPECT_TRUE(RL_TEST_FAULT("chosen"));
    EXPECT_FALSE(RL_TEST_FAULT("other"));
    RandLAPACK::test_hooks::fault = nullptr;
    EXPECT_FALSE(RL_TEST_FAULT("chosen"));
}
