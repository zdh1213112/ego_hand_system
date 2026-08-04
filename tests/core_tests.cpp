#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "ego_hand/session.hpp"

int main() {
    const std::vector<std::uint64_t> left{1000, 2000, 3000, 4000};
    const std::vector<std::uint64_t> right{1990, 3010, 4015};
    const auto pairs = ego_hand::pairTimestamps(left, right, 20);
    assert(pairs.size() == 3);
    assert(pairs[0].left_index == 1 && pairs[0].right_index == 0);
    assert(pairs[0].deltaUs() == -10);
    const auto statistics = ego_hand::calculatePairingStatistics(left, right, pairs);
    assert(statistics.skipped_left == 1);
    assert(statistics.skipped_right == 0);
    assert(statistics.max_abs_delta_us == 15);
    std::cout << "core tests passed\n";
    return 0;
}

