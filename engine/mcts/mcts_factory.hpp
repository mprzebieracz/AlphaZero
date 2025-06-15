#ifndef MCTSFACTORY_HPP
#define MCTSFACTORY_HPP

#include "mcts.hpp"
#include <inferer.hpp>
#include <memory>

class MCTSFactory {
  private:
    InfererFactory &inferer_factory;
    float c_init;
    float c_base;
    float eps;
    float alpha;

  public:
    MCTSFactory(InfererFactory &inferer_factory, float c_init = 1.25,
                float c_base = 19652, float eps = 0.25, float alpha = 0.3);

    std::unique_ptr<MCTS> get_mcts();
};

#endif // MCTSFACTORY_HPP
