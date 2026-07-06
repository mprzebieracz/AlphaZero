#ifndef MCTS_HPP
#define MCTS_HPP

#include "game.hpp"
#include "inferer.hpp"
#include <cstddef>
#include <memory>
#include <memory_resource>
#include <torch/torch.h>
#include <vector>

// The arena is reset (pool.release()) at the start of every search(); nodes store
// children sparsely (one entry per legal action), so even chess searches with
// thousands of simulations stay well within this.
constexpr size_t default_arena_size_in_bytes = static_cast<const size_t>(64) * 1024 * 1024;

class MCTS {
    using InfererPtr = std::unique_ptr<Inferer>;
    InfererPtr network;
    float c_init;
    float c_base;
    float eps;
    float alpha;
    torch::Device device;

    std::vector<std::byte> arena_buffer;
    std::pmr::monotonic_buffer_resource pool;

  public:
    MCTS(std::unique_ptr<Inferer> &&network, float c_init = 1.25f, float c_base = 19652.0f,
         float eps = 0.25f, float alpha = 0.3f,
         size_t arena_size_bytes = default_arena_size_in_bytes);

    MCTS(std::string network_path, torch::Device device, float c_init = 1.25f,
         float c_base = 19652.0f, float eps = 0.25f, float alpha = 0.3f,
         size_t arena_size_bytes = default_arena_size_in_bytes);

    std::pair<std::vector<float>, float> search(const Game &game, int num_simulations = 800,
                                                int batch_size = 8);

  private:
    class Node;
    void evaluate_batch(std::vector<std::pair<Node *, std::shared_ptr<Game>>> &leaves,
                        std::pmr::memory_resource *pool);

    std::vector<std::pair<int, float>> get_policy_from_logits(const inference_result &res,
                                                              bool dirichletNoise = false) const;

    static std::vector<float> sample_dirichlet(const std::vector<float> &alpha);
};

#endif // MCTS_HPP
