#include "mcts.hpp"
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
                float c_base = 19652, float eps = 0.25, float alpha = 0.3)
        : inferer_factory(inferer_factory), c_init(c_init), c_base(c_base),
          eps(eps), alpha(alpha) {}

    std::unique_ptr<MCTS> get_mcts() {
        return std::make_unique<MCTS>(inferer_factory.get_inferer(), c_init,
                                      c_base, eps, alpha);
    }
};
