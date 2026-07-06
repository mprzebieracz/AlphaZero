#include "game/game.hpp"
#include "mcts.hpp"
#include "replay_buffer.hpp"
#include <c10/core/Device.h>
#include <game/chess.hpp>
#include <game/connect4.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::unique_ptr<T>)

PYBIND11_MODULE(engine_bind, m) {
    try {
        py::class_<Transition>(m, "Transition")
            .def(py::init<torch::Tensor, torch::Tensor, float>())
            .def_readwrite("state", &Transition::state)
            .def_readwrite("policy", &Transition::policy)
            .def_readwrite("reward", &Transition::reward);

        py::class_<ReplayBuffer>(m, "ReplayBuffer")
            .def(py::init<size_t>())
            .def("add", &ReplayBuffer::add)
            .def("sample", &ReplayBuffer::sample)
            .def("get_size", &ReplayBuffer::get_size);

        py::class_<Game, std::shared_ptr<Game>>(m, "Game")
            .def("get_legal_actions", &Game::get_legal_actions)
            .def("step", &Game::step)
            .def("reset", &Game::reset)
            .def("render", &Game::render)
            .def("get_action_size", &Game::getActionSize)
            .def("canonical_state",
                 [](const Game &g) {
                     torch::Tensor t = torch::empty(g.get_state_shape(), torch::kFloat32);
                     g.write_canonical_state(t.data_ptr<float>());
                     return t;
                 })
            .def_property_readonly("is_terminal", &Game::is_terminal)
            .def_property_readonly("current_player", &Game::get_current_player)
            .def_property_readonly("reward", &Game::reward);

        py::class_<Game2D<6, 7>, Game, std::shared_ptr<Game2D<6, 7>>>(m, "Game2D_6_7")
            .def("get_board_state", &Game2D<6, 7>::get_board_state)
            .def_readonly_static("ROWS", &Game2D<6, 7>::ROWS)
            .def_readonly_static("COLS", &Game2D<6, 7>::COLS);

        py::class_<Connect4, Game2D<6, 7>, std::shared_ptr<Connect4>>(m, "Connect4")
            .def(py::init<>())
            .def(py::init<const std::vector<std::vector<int>> &>(), py::arg("initial_board"))
            .def_readonly_static("action_dim", &Connect4::action_dim)
            .def_property_readonly_static("state_dim",
                                          [](py::object /* self */) { return Connect4::state_dim; });

        py::class_<Game2D<8, 8>, Game, std::shared_ptr<Game2D<8, 8>>>(m, "Game2D_8_8")
            .def("get_board_state", &Game2D<8, 8>::get_board_state)
            .def_readonly_static("ROWS", &Game2D<8, 8>::ROWS)
            .def_readonly_static("COLS", &Game2D<8, 8>::COLS);

        py::class_<Chess, Game2D<8, 8>, std::shared_ptr<Chess>>(m, "Chess")
            .def(py::init<>())
            .def("set_custom_state", &Chess::set_custom_state, py::arg("board"),
                 py::arg("active_player"), py::arg("en_passant_col") = -1, py::arg("k_mc") = 0,
                 py::arg("r1_mc") = 0, py::arg("r2_mc") = 0, py::arg("K_mc") = 0,
                 py::arg("R1_mc") = 0, py::arg("R2_mc") = 0)
            .def_readonly_static("action_dim", &Chess::action_dim)
            .def_property_readonly_static(
                "state_dim",
                [](py::object /* self */) {
                    return std::make_tuple(Chess::state_dim[0], Chess::state_dim[1],
                                           Chess::state_dim[2]);
                })
            // (from_row, from_col, to_row, to_col, promotion) <-> action index
            .def_static("decode_action",
                        [](int action) {
                            auto a = Chess::decode_action(action);
                            return std::make_tuple(a.r1, a.c1, a.r2, a.c2, a.promotion);
                        })
            .def_static("encode_action", [](int r1, int c1, int r2, int c2, int promotion) {
                return Chess::encode_action(ChessAction<>(r1, c1, r2, c2, promotion));
            });

        py::class_<MCTS>(m, "MCTS")
            .def(py::init<std::string, torch::Device, float, float, float, float>(),
                 py::arg("network_path"), py::arg("device"), py::arg("c_init") = 1.25f,
                 py::arg("c_base") = 19652.0f, py::arg("eps") = 0.25f, py::arg("alpha") = 0.3f)
            .def("search", &MCTS::search, py::arg("game"), py::arg("num_simulations") = 800,
                 py::arg("batch_size") = 32);

    } catch (const std::exception &e) {
        py::print("Exception during binding:", e.what());
        throw;
    }
}
