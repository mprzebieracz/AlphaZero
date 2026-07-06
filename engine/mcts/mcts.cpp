#include "mcts.hpp"
#include "basic_inferer.hpp"

#include <algorithm>
#include <c10/core/Device.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <torch/script.h>
#include <unordered_map>

struct MCTS::Node {
    // Children are stored sparsely: children[k] is the child for action
    // valid_actions[k]. A dense array indexed by action would waste ~160KB per
    // expansion for chess (action space 20480, ~35 legal moves per position).
    Node **children = nullptr;
    int *valid_actions = nullptr;
    int valid_action_count = 0;
    Node *parent = nullptr;
    int visits = 0;
    float value = 0.0f;
    float prior = 0.0f;
    bool expanded = false;
    bool is_terminal = false;
    float reward = 0.0f;
    int virtual_loss_count = 0;
    static constexpr float VL = 1.0f;

    Node(float prior_value = 0.0f, Node *parent_node = nullptr, bool terminal = false,
         float rew = 0.0f);

    float Q() const;
    float UCB(float exploration_weight) const;

    void expand(const std::vector<std::pair<int, float>> &policy, std::pmr::memory_resource *pool);

    bool terminal() const;
    bool is_expanded() const;

    std::pair<int, Node *> select_child(float exploration_weight) const;

    static void backpropagate(Node *node, float value, bool vloss = false);
};

void MCTS::Node::backpropagate(Node *node, float value, bool vloss) {
    while (node != nullptr) {
        node->visits++;
        node->value += value;
        if (vloss)
            node->virtual_loss_count--;
        node = node->parent;
        value = -value;
    }
}

MCTS::Node::Node(float prior_value, Node *parent_node, bool terminal, float rew)
    : parent(parent_node), prior(prior_value), is_terminal(terminal), reward(rew) {}

float MCTS::Node::Q() const {
    // Q is from this node's own (player-to-move) perspective; the parent selects on
    // -Q(child). Virtual loss must make an in-flight node look *worse* to the
    // parent, i.e. push -Q down, so it is *added* here. (Subtracting it - the old
    // bug - made concurrent simulations pile onto the same leaf.)
    float adj_value = value + (virtual_loss_count * VL);
    int adj_visits = visits + virtual_loss_count;
    return adj_visits > 0 ? adj_value / adj_visits : 0.0f;
}

float MCTS::Node::UCB(float exploration_weight) const {
    if (parent == nullptr)
        return 0.0f;
    return -Q() + (exploration_weight * prior *
                   std::sqrt(parent->visits + parent->virtual_loss_count + 1) /
                   (1 + visits + virtual_loss_count));
}

void MCTS::Node::expand(const std::vector<std::pair<int, float>> &policy,
                        std::pmr::memory_resource *pool) {
    children = static_cast<Node **>(pool->allocate(policy.size() * sizeof(Node *), alignof(Node *)));
    valid_actions = static_cast<int *>(pool->allocate(policy.size() * sizeof(int), alignof(int)));

    std::pmr::polymorphic_allocator<Node> alloc(pool);
    int idx = 0;
    for (const auto &[action, p] : policy) {
        if (p == 0.0f)
            continue;
        Node *child = alloc.allocate(1);
        alloc.construct(child, p, this, false, 0.0f);
        children[idx] = child;
        valid_actions[idx] = action;
        ++idx;
    }
    valid_action_count = idx;
    expanded = true;
}

bool MCTS::Node::terminal() const {
    return is_terminal;
}

bool MCTS::Node::is_expanded() const {
    return expanded;
}

std::pair<int, MCTS::Node *> MCTS::Node::select_child(float exploration_weight) const {
    int best_index = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < valid_action_count; ++k) {
        float ucb_value = children[k]->UCB(exploration_weight);
        if (ucb_value > best_value) {
            best_value = ucb_value;
            best_index = k;
        }
    }
    return {best_index != -1 ? valid_actions[best_index] : -1,
            best_index != -1 ? children[best_index] : nullptr};
}

MCTS::MCTS(std::unique_ptr<Inferer> &&network_ptr, float c_init, float c_base, float eps,
           float alpha, size_t arena_size_bytes)
    : network(std::move(network_ptr)), c_init(c_init), c_base(c_base), eps(eps), alpha(alpha),
      device(this->network->device), arena_buffer(arena_size_bytes),
      pool(arena_buffer.data(), arena_buffer.size()) {}

MCTS::MCTS(std::string network_path, torch::Device device, float c_init, float c_base, float eps,
           float alpha, size_t arena_size_bytes)
    : network([&device, &network_path]() {
          auto network_inferer_factory = NetworkInfererFactory(network_path, device);
          return network_inferer_factory.get_inferer();
      }()),
      c_init(c_init), c_base(c_base), eps(eps), alpha(alpha), device(this->network->device),
      arena_buffer(arena_size_bytes), pool(arena_buffer.data(), arena_buffer.size()) {}

void MCTS::evaluate_batch(std::vector<std::pair<Node *, std::shared_ptr<Game>>> &leaves,
                          std::pmr::memory_resource *pool) {
    if (leaves.empty())
        return;

    std::vector<const GameState *> states;
    std::vector<size_t> result_index(leaves.size());
    std::unordered_map<Node *, size_t> seen;
    seen.reserve(leaves.size());
    for (size_t i = 0; i < leaves.size(); ++i) {
        Node *node = leaves[i].first;
        auto [it, inserted] = seen.try_emplace(node, states.size());
        result_index[i] = it->second;
        if (inserted) {
            states.push_back(leaves[i].second->get_canonical_state().get());
        }
    }

    auto outputs = network->infer(states);

    for (size_t i = 0; i < leaves.size(); ++i) {
        Node *node = leaves[i].first;
        const auto &res = outputs[result_index[i]];

        if (node->is_expanded()) {
            Node::backpropagate(node, res.value, true);
            continue;
        }

        auto policy = get_policy_from_logits(res, false);
        node->expand(policy, pool);
        Node::backpropagate(node, res.value, true);
    }
}

std::pair<std::vector<float>, float> MCTS::search(const Game &game, int num_simulations,
                                                  int batch_size) {
    pool.release();
    Node root_node(0.0f, nullptr, game.is_terminal(), game.reward());

    auto inference_res =
        network->infer(std::vector<const GameState *>{game.get_canonical_state().get()});
    float root_value = inference_res.front().value;
    auto p_init = get_policy_from_logits(inference_res.front(), true);
    root_node.expand(p_init, &pool);

    int simulations_done = 0;
    std::vector<std::pair<Node *, std::shared_ptr<Game>>> leaves;
    while (simulations_done < num_simulations) {
        leaves.clear();

        for (int b = 0; b < batch_size && simulations_done < num_simulations;
             ++b, ++simulations_done) {
            Node *node = &root_node;
            auto current_game = game.clone();
            float c_puct = std::log((1 + node->visits + c_base) / c_base) + c_init;

            while (node->is_expanded() && !node->terminal()) {
                auto [best_action, best_child] = node->select_child(c_puct);
                if (best_child == nullptr)
                    break;
                node->virtual_loss_count++;
                node = best_child;
                current_game->step(best_action);

                if (node->visits == 0 && current_game->is_terminal()) {
                    node->is_terminal = true;
                    node->reward = current_game->reward();
                }
                c_puct = std::log((1 + node->visits + c_base) / c_base) + c_init;
            }

            node->virtual_loss_count++;

            if (!node->terminal()) {
                leaves.emplace_back(node, std::move(current_game));
            } else {
                // node->reward is already expressed for the player to move at this
                // node (Game::reward()'s contract), the same convention in which
                // evaluate_batch() passes res.value - no re-negation.
                Node::backpropagate(node, node->reward, true);
            }
        }

        evaluate_batch(leaves, &pool);
    }

    std::vector<float> pi(game.getActionSize(), 0.0f);
    for (int k = 0; k < root_node.valid_action_count; ++k) {
        pi[root_node.valid_actions[k]] = static_cast<float>(root_node.children[k]->visits);
    }
    float sum = std::accumulate(pi.begin(), pi.end(), 0.0f);
    if (sum > 0.0f)
        for (auto &x : pi)
            x /= sum;
    return {pi, root_value};
}

std::vector<std::pair<int, float>> MCTS::get_policy_from_logits(const inference_result &res,
                                                                bool dirichletNoise) const {
    const auto &legal_actions = res.legal_actions;
    std::vector<std::pair<int, float>> policy_vec;
    policy_vec.reserve(legal_actions.size());

    float max_logit = -std::numeric_limits<float>::infinity();
    for (float logit : res.legal_action_logits) {
        max_logit = std::max(logit, max_logit);
    }

    float sum_exp = 0.0f;
    for (size_t j = 0; j < legal_actions.size(); ++j) {
        float p = std::exp(res.legal_action_logits[j] - max_logit);
        policy_vec.emplace_back(legal_actions[j], p);
        sum_exp += p;
    }

    if (sum_exp > 0.0f) {
        for (auto &pair : policy_vec) {
            pair.second /= sum_exp;
        }
    }

    if (dirichletNoise) {
        std::vector<float> alpha_vec(legal_actions.size(), alpha);
        auto noise_vec = sample_dirichlet(alpha_vec);

        for (size_t i = 0; i < policy_vec.size(); ++i) {
            policy_vec[i].second = ((1.0f - eps) * policy_vec[i].second) + (eps * noise_vec[i]);
        }
    }

    return policy_vec;
}

std::vector<float> MCTS::sample_dirichlet(const std::vector<float> &alpha) {
    static thread_local std::mt19937 gen{std::random_device{}()};
    std::vector<float> x(alpha.size());
    float sum = 0.0f;
    for (size_t i = 0; i < alpha.size(); ++i) {
        std::gamma_distribution<float> dist(alpha[i], 1.0f);
        x[i] = dist(gen);
        sum += x[i];
    }
    if (sum > 0.0f) {
        for (auto &v : x)
            v /= sum;
    }
    return x;
}
