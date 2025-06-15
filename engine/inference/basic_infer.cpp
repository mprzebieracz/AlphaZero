#include "network.hpp"
#include <game/connect4.hpp>
#include <torch/torch.h> // For torch::Tensor and device

#include "basic_inferer.hpp"

NetworkInferer::NetworkInferer(const std::string &network_file_path,
                               torch::Device device, int resblock_filter_size,
                               int residual_block_count)
    : network(std::get<0>(Connect4::state_dim),
              std::get<1>(Connect4::state_dim),
              std::get<2>(Connect4::state_dim), residual_block_count,
              Connect4::action_dim, resblock_filter_size),
      Inferer(device) {
    // Load the network weights here if needed, e.g.:
    // TODO: come back to this
    // torch::load(network, network_file_path.data());
    network->to(device);
    network->eval();
}

// infer method implementation
std::pair<torch::Tensor, float>
NetworkInferer::infer(torch::Tensor game_state_tensor) {
    auto [policy, eval] = network->forward(game_state_tensor);
    return std::make_pair(std::move(policy), eval.item<float>());
}

// Constructor for NetworkInfererFactory
NetworkInfererFactory::NetworkInfererFactory(
    const std::string &network_file_path, torch::Device device)
    : _network_file_path(network_file_path), _device(device) {}

// get_inferer method implementation
std::unique_ptr<Inferer> NetworkInfererFactory::get_inferer() {
    return std::make_unique<NetworkInferer>(_network_file_path, _device);
}
