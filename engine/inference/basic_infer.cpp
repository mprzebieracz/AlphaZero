#include "basic_inferer.hpp"
#include "inferer.hpp"
#include <connect4.hpp>
#include <torch/csrc/jit/serialization/import.h>

#include <torch/torch.h> // For torch::Tensor and device

NetworkInferer::NetworkInferer(const std::string &network_file_path,
                               torch::Device device, int resblock_filter_size,
                               int residual_block_count)
    : Inferer(device), network(torch::jit::load(network_file_path, device)),
      infer_method(network.get_method("infer")) {

    // torch::OrderedDict<std::string, torch::Tensor> weights;
    // torch::load(weights, network_file_path);
    // network->load_state_dict(weights);
    // Load the network weights here if needed, e.g.:
    // TODO: come back to this
    // torch::load(network, network_file_path);

    network.to(device);
    network.eval();
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
    : _network_file_path(network_file_path), _device(device) {}

// get_inferer method implementation
std::unique_ptr<Inferer> NetworkInfererFactory::get_inferer() {
    return std::make_unique<NetworkInferer>(_network_file_path, _device);
}
