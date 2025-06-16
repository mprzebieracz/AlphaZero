#ifndef MCTS_HPP
#define MCTS_HPP

#include "game.hpp"
#include "inferer.hpp"
#include <memory>
#include <torch/torch.h>
#include <vector>

using std::make_unique;
using std::unique_ptr;
using std::vector;

struct Node {
    using NodePtr = std::unique_ptr<Node>;
    unique_ptr<Game> game_state;
    vector<NodePtr> children;
    Node *parent = nullptr;
    int visits = 0;
    float value = 0.0f;
    float prior = 0.0f;
    bool expanded = false;

    Node(std::unique_ptr<Game> state, float prior_value = 0.0f,
         Node *parent_node = nullptr);

    float Q() const;
    float UCB(float exploration_weight) const;

    void expand(const std::vector<float> &policy);

    bool is_terminal() const;
    bool isExpanded() const;

    std::pair<int, Node *> selectChild(float exploration_weight) const;
};

class MCTS {
    using InfererPtr = unique_ptr<Inferer>;
    InfererPtr network;
    float c_init;
    float c_base;
    float eps;
    float alpha;
    torch::Device device;

  public:
    MCTS(unique_ptr<Inferer> &&network, float c_init = 1.25f,
         float c_base = 19652.0f, float eps = 0.25f, float alpha = 0.3f);

    MCTS(std::string network_path, torch::Device device, float c_init = 1.25f,
         float c_base = 19652.0f, float eps = 0.25f, float alpha = 0.3f);

    std::vector<float> search(const Game &game, int num_simulations = 800);

  private:
    std::pair<std::vector<float>, float>
    get_policy_value(const Game &game, bool dirichletNoise = false);

    void backpropagate(Node *node, float value);

    static std::vector<float> sample_dirichlet(const std::vector<float> &alpha);
};

#endif // MCTS_HPP
