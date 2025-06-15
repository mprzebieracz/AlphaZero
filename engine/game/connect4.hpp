#ifndef CONNECT_4_HPP
#define CONNECT_4_HPP

#include "game.hpp"
#include <iostream>
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

    Connect4() {
        reset();
    }
    ~Connect4() override = default;

    void reset() override {
        board = vector<vector<int>>(ROWS, vector<int>(COLS, 0));
        currentPlayer = 1; // Player 1 starts
        finished = false;
        _reward = 0.0f;
    }

    int getActionSize() const override {
        return COLS; // 7 columns to drop a piece
    }

    vector<int> getLegalActions() const override {
        vector<int> legalActions;
        for (int col = 0; col < COLS; col++) {
            if (board[0][col] == 0) { // Check if the column is not full
                legalActions.push_back(col);
            }
        }
        return legalActions;
    }

    void step(int action) override {
        if (finished || action < 0 || action >= COLS || board[0][action] != 0) {
            throw std::invalid_argument("Invalid action");
        }

        int placedRow = -1;
        int placedCol = action;
        // Drop the piece in the selected column
        for (int row = ROWS - 1; row >= 0; row--) {
            if (board[row][action] == 0) {
                board[row][action] = currentPlayer;
                placedRow = row;
                break;
            }
        }

        if (checkWin(placedRow, placedCol)) {
            finished = true;
            _reward = currentPlayer;
        } else if (getLegalActions().empty()) {
            finished = true; // Draw
            _reward = 0.0f;
        } else {
            currentPlayer = -currentPlayer; // Switch player
        }
    }

    bool is_terminal() const override {
        return finished;
    }

    float reward() const override {
        return _reward;
    }

    torch::Tensor get_canonical_state() const override {
        // Convert the board to a tensor with shape (1, 1, ROWS, COLS)
        torch::Tensor state = torch::zeros({1, 1, ROWS, COLS}, torch::kFloat32);
        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                state[0][0][row][col] = board[row][col];
            }
        }
        return state;
    }

    std::unique_ptr<Game> clone() const override {
        auto newGame = std::make_unique<Connect4>();
        newGame->board = board;
        newGame->currentPlayer = currentPlayer;
        newGame->finished = finished;
        newGame->_reward = _reward;
        return newGame;
    }

    void render() const override {
        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                std::cout << board[row][col] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Current Player: " << currentPlayer << std::endl;
    }

  private:
    bool checkWin(int row, int col) const {
        // Check horizontal, vertical, and diagonal directions for a win
        return checkDirection(row, col, 1, 0) || // Horizontal
               checkDirection(row, col, 0, 1) || // Vertical
               checkDirection(row, col, 1, 1) || // Diagonal /
               checkDirection(row, col, 1, -1);  // Diagonal
    }

    bool checkDirection(int row, int col, int dRow, int dCol) const {
        int count = 0;
        for (int i = -3; i <= 3; i++) {
            int r = row + i * dRow;
            int c = col + i * dCol;
            if (r >= 0 && r < ROWS && c >= 0 && c < COLS &&
                board[r][c] == currentPlayer) {
                count++;
                if (count == 4)
                    return true; // Found a winning line
            } else {
                count = 0; // Reset count if not matching
            }
        }
        return false;
    }

    vector<vector<int>> board;

    int currentPlayer;
    bool finished;
    float _reward;
};

#endif // !CONNECT_4_HPP
