# 1. 规定 CMake 的最低版本
cmake_minimum_required(VERSION 3.10)

# 2. 项目名称
project(EggNetProject)

# 3. 指定 C++ 标准为 20 (因为你用了 std::complex 等新特性)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4. 搜集所有的源文件 (.cpp)
# 注意：.hpp 不需要写进来，因为 .cpp 里的 #include 会自动找它们
add_executable(number_problem 
    number_problem.cpp 
    data_loader.cpp
)

# 新增实战推理程序！
add_executable(predict predict.cpp data_loader.cpp)
target_compile_options(predict PRIVATE -O2)

# 5. (可选) 如果你追求极致速度，开启 O2 优化
if(MSVC)
    target_compile_options(number_problem PRIVATE /O2)
else()
    target_compile_options(number_problem PRIVATE -O2)
endif()