#include "basic_inferer.hpp"
#include "inferer.hpp"
#include <connect4.hpp>
#include <mutex>

#include <torch/torch.h> // For torch::Tensor and device

NetworkInferer::NetworkInferer(std::shared_ptr<Network> network,
                               torch::Device device)
    : Inferer(device), network(network),
      infer_method(network->get_method("infer")) {
    // Load the network weights here if needed, e.g.:
    // TODO: come back to this
    // torch::load(network, network_file_path);

    // network.to(device);
    // network.eval();
    // torch::jit::script::Module module = torch::jit::load(network_file_path);
}

// infer method implementation
std::pair<torch::Tensor, float>
NetworkInferer::infer(torch::Tensor game_state_tensor) {
    auto result = infer_method({game_state_tensor});
    auto outputs = result.toTuple()->elements();
    torch::Tensor policy = outputs[0].toTensor();
    torch::Tensor value = outputs[1].toTensor();

    return std::make_pair(std::move(policy), value.item<float>());
}

// Constructor for NetworkInfererFactory
NetworkInfererFactory::NetworkInfererFactory(
    const std::string &network_file_path, torch::Device device)
    : network_file_path(network_file_path), device(device),
      network(std::make_shared<Network>(
          torch::jit::load(network_file_path, device))) {
    network->to(device);
    network->eval();
}

// get_inferer method implementation
std::unique_ptr<Inferer> NetworkInfererFactory::get_inferer() {
    auto lock_guard = std::lock_guard(get_inferer_mutex);
    return std::make_unique<NetworkInferer>(network, device);
}
