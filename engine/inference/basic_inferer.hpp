#ifndef NETWORK_INFERER_HPP
#define NETWORK_INFERER_HPP

#include "mcts.hpp"
#include "network.hpp"
#include <game/connect4.hpp>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <utility>

// Forward declaration of AlphaZeroNetworkImpl (if needed)
// class AlphaZeroNetworkImpl;

class NetworkInferer : public Inferer {
  private:
    AlphaZeroNetwork network;

  public:
    NetworkInferer(const std::string &network_file_path, torch::Device device,
                   int resblock_filter_size = 64,
                   int residual_block_count = 10);

    std::pair<torch::Tensor, float>
    infer(torch::Tensor game_state_tensor) override;
    // Optional: expose device if needed
    // torch::Device device() const;
};

class NetworkInfererFactory : public InfererFactory {
  private:
    std::string _network_file_path;
    torch::Device _device;

  public:
    NetworkInfererFactory(const std::string &network_file_path,
                          torch::Device device);

    std::unique_ptr<Inferer> get_inferer() override;
};

#endif // NETWORK_INFERER_HPP
