#include "mcts.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>

#include <random>

Node::Node(std::unique_ptr<Game> state, float prior_value, Node *parent_node)
    : game_state(std::move(state)), prior(prior_value), parent(parent_node) {
    children.resize(game_state->getActionSize());
}

float Node::Q() const {
    return visits > 0 ? value / visits : 0.0f;
}

float Node::UCB(float exploration_weight) const {
    if (!parent)
        return 0;
    return Q() + exploration_weight * prior * std::sqrt(parent->visits) /
                     (1 + visits);
}

void Node::expand(const std::vector<float> &policy) {
    for (int i = 0; i < policy.size(); i++) {
        if (policy[i] == 0.0f)
            continue;
        auto child_state = game_state->clone();
        child_state->step(i);

        children[i] =
            make_unique<Node>(std::move(child_state), policy[i], this);
    }
    expanded = true;
}

bool Node::is_terminal() const {
    return game_state->is_terminal();
}

bool Node::isExpanded() const {
    return expanded;
}

std::pair<int, Node *> Node::selectChild(float exploration_weight) const {
    int best_index = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < children.size(); i++) {
        if (!children[i])
            continue;
        float ucb_value = children[i]->UCB(exploration_weight);
        if (ucb_value > best_value) {
            best_value = ucb_value;
            best_index = i;
        }
    }
    return {best_index, children[best_index].get()};
}

MCTS::MCTS(unique_ptr<Inferer> &&network, float c_init, float c_base, float eps,
           float alpha)
    : network(std::move(network)), c_init(c_init), c_base(c_base), eps(eps),
      alpha(alpha), device(this->network->device) {}

std::vector<float> MCTS::search(const Game &game, int num_simulations) {
    auto root = game.clone();
    Node root_node = Node(std::move(root));

    auto [p_init, _] = get_policy_value(game, true);
    root_node.expand(p_init);

    for (int sim = 0; sim < num_simulations; sim++) {
        Node *node = &root_node;
        float c_puct = std::log((1 + node->visits + c_base) / c_base) + c_init;

        // Selection phase
        while (node->isExpanded() && !node->is_terminal()) {
            auto [best_action, best_child] = node->selectChild(c_puct);
            node = best_child;
            c_puct = std::log((1 + node->visits + c_base) / c_base) + c_init;
        }

        float value;
        if (!node->is_terminal()) {
            auto [p, v] = get_policy_value(*node->game_state, false);
            node->expand(p);
            value = v;
        } else
            value = node->game_state->reward();

        backpropagate(node, value);
    }

    int A = game.getActionSize();
    std::vector<float> policy(A, 0.0f);
    std::vector<float> pi(A, 0.0f);
    for (int a = 0; a < A; a++) {
        if (root_node.children[a])
            pi[a] = static_cast<float>(root_node.children[a]->visits);
    }
    float sum = std::accumulate(pi.begin(), pi.end(), 0.0f);
    if (sum > 0.0f)
        for (auto &x : pi)
            x /= sum;
    return pi;
}

std::pair<std::vector<float>, float>
MCTS::get_policy_value(const Game &game, bool dirichletNoise) {
    torch::NoGradGuard no_grad;
    auto state_tensor = game.get_canonical_state();
    if (state_tensor.device() != device)
        state_tensor = state_tensor.to(device);

    auto [policy_logits, value] = network->infer(state_tensor);
    policy_logits = policy_logits.squeeze(0);

    int A = game.getActionSize();
    torch::Tensor policy = torch::softmax(policy_logits, 0);
    if (dirichletNoise) {
        std::vector<float> alpha_vec(A, alpha);
        auto noise_vec = sample_dirichlet(alpha_vec);
        auto noise_t = torch::tensor(
            noise_vec,
            torch::TensorOptions().dtype(torch::kFloat32).device(device));
        policy = (1.0f - eps) * policy + eps * noise_t;
    }

    auto legal = game.getLegalActions();
    std::vector<float> mask_vec(A, 0.0f);
    for (int a : legal)
        mask_vec[a] = 1.0f;
    auto mask_t = torch::tensor(
        mask_vec, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    policy = policy * mask_t;

    float sum = policy.sum().item<float>();

    policy = policy.cpu();

    std::vector<float> policy_vec(A, 0.0f);
    if (sum > 0.0f) {
        policy = policy / sum;
        std::memcpy(policy_vec.data(), policy.data_ptr<float>(),
                    A * sizeof(float));
    } else {
        float inv = 1.0f / static_cast<float>(legal.size());
        for (int a : legal)
            policy_vec[a] = inv;
    }

    return {policy_vec, value};
}

void MCTS::backpropagate(Node *node, float value) {
    while (node) {
        node->visits++;
        node->value += value;
        node = node->parent;
        value = -value;
    }
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
