cmake_minimum_required(VERSION 3.10)
project(MyTorchProject LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set path to LibTorch
set(Torch_DIR "$ENV{HOME}/libs/libtorch/share/cmake/Torch")  # CMake config path

# Find LibTorch
find_package(Torch REQUIRED)

# Workaround for missing nvToolsExt
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)
    # Prevent link errors due to missing .so
    set_target_properties(CUDA::nvToolsExt PROPERTIES
        IMPORTED_LOCATION ""
    )
endif()



# Enable RPATH for local library path
set(CMAKE_BUILD_RPATH "$ENV{HOME}/libs/libtorch/lib")

# Add the executable
add_executable(main main.cpp)

# Link against LibTorch
target_link_libraries(main "${TORCH_LIBRARIES}")

# Enable CUDA if available (optional, just for clarity)
if (TORCH_CUDA_FOUND)
    message(STATUS "Compiling with CUDA support")
else()
    message(STATUS "Compiling without CUDA support")
endif()

# Set this flag if LibTorch was built with C++11 ABI
target_compile_definitions(main PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
