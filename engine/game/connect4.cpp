#include "connect4.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

Connect4::Connect4() : board(), currentPlayer(1), finished(false), _reward(0.0f) {}

static int determine_current_player(const Connect4::board_t &board) {
    int player1_count = 0;
    int player2_count = 0;
    for (const auto &row : board) {
        for (auto cell : row) {
            if (cell == 1)
                player1_count++;
            else if (cell == -1)
                player2_count++;
        }
    }
    return (player1_count == player2_count) ? 1 : -1;
}

static Connect4::board_t vector_to_board_t(const std::vector<std::vector<int>> &vec) {
    if (vec.size() != Connect4::ROWS) {
        throw std::invalid_argument("Board must have " + std::to_string(Connect4::ROWS) + " rows");
    }
    Connect4::board_t b = {};
    for (int r = 0; r < Connect4::ROWS; r++) {
        if (vec[r].size() != Connect4::COLS) {
            throw std::invalid_argument("Board must have " + std::to_string(Connect4::COLS) +
                                        " columns");
        }
        for (int c = 0; c < Connect4::COLS; c++) {
            b[r][c] = vec[r][c];
        }
    }
    return b;
}

Connect4::Connect4(const std::vector<std::vector<int>> &initial_board)
    : Connect4(vector_to_board_t(initial_board)) {}

Connect4::Connect4(const board_t &initial_board)
    : board(initial_board), currentPlayer(determine_current_player(initial_board)), finished(false),
      _reward(0.0f) {
    if (hasWin(board, currentPlayer)) {
        finished = true;
        _reward = 1.0f;
    } else if (hasWin(board, -currentPlayer)) {
        finished = true;
        _reward = -1.0f;
    } else if (isBoardFull(board)) {
        finished = true;
    }
}

bool Connect4::hasWin(const board_t &board, int player) {
    auto check_dir = [&](int row, int col, int dRow, int dCol) {
        int count = 0;
        for (int i = 0; i < 4; i++) {
            int r = row + (i * dRow);
            int c = col + (i * dCol);
            if (r >= 0 && r < ROWS && c >= 0 && c < COLS && board[r][c] == player) {
                count++;
            }
        }
        return count == 4;
    };

    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS; col++) {
            if (board[row][col] != player)
                continue;
            if (check_dir(row, col, 0, 1) || check_dir(row, col, 1, 0) ||
                check_dir(row, col, 1, 1) || check_dir(row, col, -1, 1)) {
                return true;
            }
        }
    }
    return false;
}

bool Connect4::isBoardFull(const board_t &board) {
    for (int col = 0; col < COLS; col++) {
        if (board[0][col] == 0)
            return false;
    }
    return true;
}

void Connect4::reset() {
    reset_initial_state();
}

int Connect4::getActionSize() const {
    return COLS;
}

std::vector<int> Connect4::get_legal_actions() const {
    std::vector<int> legalActions;
    for (int col = 0; col < COLS; col++) {
        if (board[0][col] == 0) {
            legalActions.push_back(col);
        }
    }
    return legalActions;
}

void Connect4::step(int action) {
    if (finished || action < 0 || action >= COLS || board[0][action] != 0) {
        throw std::invalid_argument("Invalid action " + std::to_string(finished) + " " +
                                    std::to_string(action));
    }

    int placedRow = -1;
    int placedCol = action;
    for (int row = ROWS - 1; row >= 0; row--) {
        if (board[row][action] == 0) {
            board[row][action] = currentPlayer;
            placedRow = row;
            break;
        }
    }

    if (checkWin(placedRow, placedCol)) {
        finished = true;
        _reward = 1.0f;
    } else if (get_legal_actions().empty()) {
        finished = true;
        _reward = 0.0f;
    } else {
        currentPlayer = -currentPlayer;
    }
}

bool Connect4::is_terminal() const {
    return finished;
}

int Connect4::get_current_player() const {
    return currentPlayer;
}

float Connect4::reward() const {
    if (!finished)
        return 0.0f;
    return _reward;
}

std::shared_ptr<const GameState> Connect4::get_canonical_state() const {
    if (weak_from_this().expired()) {
        return {this, [](const GameState *) {}};
    }
    return shared_from_this();
}

std::vector<int64_t> Connect4::get_state_shape() const {
    return {1, ROWS, COLS};
}

void Connect4::write_canonical_state(float *out_buffer) const {
    int idx = 0;
    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS; col++) {
            out_buffer[idx++] = static_cast<float>(board[row][col] * currentPlayer);
        }
    }
}

std::shared_ptr<Game> Connect4::clone() const {
    return std::make_shared<Connect4>(*this);
}

void Connect4::render() const {
    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS; col++) {
            char piece = '.';
            switch (board[row][col]) {
            case 1:
                piece = 'X';
                break;
            case -1:
                piece = 'O';
                break;
            default:
                break;
            }
            std::cout << piece << " ";
        }
        std::cout << '\n';
    }
    std::cout << "Current Player: " << (currentPlayer == 1 ? "X" : "O") << '\n';
}

bool Connect4::checkWin(int row, int col) const {
    return checkDirection(row, col, 1, 0) || checkDirection(row, col, 0, 1) ||
           checkDirection(row, col, 1, 1) || checkDirection(row, col, 1, -1);
}

bool Connect4::checkDirection(int row, int col, int dRow, int dCol) const {
    int count = 0;
    for (int i = -3; i <= 3; i++) {
        int r = row + (i * dRow);
        int c = col + (i * dCol);
        if (r >= 0 && r < ROWS && c >= 0 && c < COLS && board[r][c] == currentPlayer) {
            count++;
            if (count == 4)
                return true;
        } else {
            count = 0;
        }
    }
    return false;
}

Connect4::board_t Connect4::get_board_state() const {
    return board;
}

void Connect4::reset_initial_state() {
    board = {};
    currentPlayer = 1;
    finished = false;
    _reward = 0.0f;
}

bool Connect4::hasWin(const std::vector<std::vector<int>> &board, int player) {
    return hasWin(vector_to_board_t(board), player);
}

bool Connect4::isBoardFull(const std::vector<std::vector<int>> &board) {
    return isBoardFull(vector_to_board_t(board));
}
