#include "test_common.hpp"
#include "utils/replay_buffer.hpp"
#include <torch/torch.h>

static Transition make_transition(float reward) {
    auto s = torch::zeros({1, 6, 7}, torch::kFloat32);
    auto p = torch::zeros({7}, torch::kFloat32);
    return Transition(s, p, reward);
}

static void test_add_and_sample_under_capacity() {
    ReplayBuffer buf(10);
    std::vector<Transition> batch;
    batch.push_back(make_transition(1.0f));
    batch.push_back(make_transition(-1.0f));
    batch.push_back(make_transition(0.0f));
    buf.add(batch);
    auto [s, p, r] = buf.sample(3);
    CHECK_EQ(s.size(0), 3);
    CHECK_EQ(p.size(0), 3);
    CHECK_EQ(r.size(0), 3);
}

static void test_capacity_wraps() {
    ReplayBuffer buf(5);
    std::vector<Transition> big;
    for (int i = 0; i < 12; ++i) big.push_back(make_transition((float)i));
    buf.add(big);
    auto [s, p, r] = buf.sample(5);
    CHECK_EQ(s.size(0), 5);
    // sample size requested > stored count is clamped to stored count.
    auto [s2, p2, r2] = buf.sample(100);
    CHECK_EQ(s2.size(0), 5);
}

static void test_sample_empty_does_not_crash() {
    ReplayBuffer buf(4);
    auto [s, p, r] = buf.sample(2);
    CHECK_EQ(s.size(0), 0);
}

int main() {
    RUN(test_add_and_sample_under_capacity);
    RUN(test_capacity_wraps);
    RUN(test_sample_empty_does_not_crash);
    std::printf("All ReplayBuffer tests passed.\n");
    return 0;
}
