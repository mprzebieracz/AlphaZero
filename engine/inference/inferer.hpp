#ifndef INFERER_H
#define INFERER_H

#include "game.hpp"
#include <torch/torch.h>

struct inference_result {
    std::vector<int> legal_actions;
    std::vector<float> legal_action_logits;
    float value;
};

struct Inferer {
    virtual std::vector<inference_result> infer(const std::vector<const GameState *> &states) = 0;
    torch::Device device;
    virtual ~Inferer() = default;

    Inferer(torch::Device device) : device(device) {}
};

class InfererFactory {
  public:
    virtual ~InfererFactory() = default;
    virtual std::unique_ptr<Inferer> get_inferer() = 0;
};

#endif // !INFERER_H
