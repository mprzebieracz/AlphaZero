g++ main.cpp -o test_libtorch \
  -I$HOME/libs/libtorch/include \
  -I$HOME/libs/libtorch/include/torch/csrc/api/include \
  -L$HOME/libs/libtorch/lib \
  -Wl,-rpath,$HOME/libs/libtorch/lib \
  -ltorch -ltorch_cpu -lc10 \
  -lc10_cuda -ltorch_cuda \
  -D_GLIBCXX_USE_CXX11_ABI=1 \
  -std=c++17
