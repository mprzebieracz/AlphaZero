#ifndef CONNECT_4_HPP
#define CONNECT_4_HPP

#include "game.hpp"
#include <memory>
#include <torch/torch.h>
#include <vector>

using std::vector;

class Connect4 : public Game {
  public:
    static constexpr int ROWS = 6;
    static constexpr int COLS = 7;
    static constexpr int action_dim = 7;
    static constexpr auto state_dim = std::make_tuple(1, 6, 7);

    Connect4();
    ~Connect4() override = default;

    void reset() override;
    int getActionSize() const override;
    vector<int> getLegalActions() const override;
    void step(int action) override;
    bool is_terminal() const override;
    float reward() const override;
    torch::Tensor get_canonical_state() const override;
    std::unique_ptr<Game> clone() const override;
    void render() const override;

  private:
    bool checkWin(int row, int col) const;
    bool checkDirection(int row, int col, int dRow, int dCol) const;

    vector<vector<int>> board;
    int currentPlayer;
    bool finished;
    float _reward;
};

#endif // CONNECT_4_HPP
