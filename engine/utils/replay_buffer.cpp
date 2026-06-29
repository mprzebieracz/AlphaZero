#include "replay_buffer.hpp"
#include <algorithm>

Transition::Transition(const torch::Tensor &s, const torch::Tensor &p, float r)
    : state(s), policy(p), reward(r) {}

ReplayBuffer::ReplayBuffer(size_t capacity)
    : capacity(capacity), buffer(capacity), rng(std::random_device{}()) {}

void ReplayBuffer::add(const std::vector<Transition> &transitions) {
    std::lock_guard<std::mutex> lock(write_mutex);
    for (const auto &transition : transitions) {
        buffer[ptr] = transition;
        ptr = (ptr + 1) % capacity;
        if (size < capacity) {
            size++;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ReplayBuffer::sample(size_t batch_size) const {
    batch_size = std::min(batch_size, size);
    if (batch_size == 0) {
        return {torch::empty({0}, torch::kFloat32),
                torch::empty({0}, torch::kFloat32),
                torch::empty({0}, torch::kFloat32)};
    }

    std::vector<torch::Tensor> states, policies;
    std::vector<float> rewards;
    states.reserve(batch_size);
    policies.reserve(batch_size);
    rewards.reserve(batch_size);

    std::uniform_int_distribution<size_t> dist(0, size - 1);
    for (size_t i = 0; i < batch_size; i++) {
        const Transition &transition = buffer[dist(rng)];
        states.push_back(transition.state.squeeze(0));
        policies.push_back(transition.policy);
        rewards.push_back(transition.reward);
    }

    return {torch::stack(states), torch::stack(policies),
            torch::tensor(rewards, torch::kFloat32)};
}
