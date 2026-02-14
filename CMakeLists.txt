cmake_minimum_required(VERSION 3.20)
project(lob_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(lob_core
  src/engine/order.cpp
  src/engine/order_book.cpp
  src/engine/matching_engine.cpp
  src/engine/simulator.cpp
  src/engine/metrics.cpp
)

target_include_directories(lob_core PUBLIC src)

target_compile_options(lob_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(lob_sim src/main.cpp)
target_link_libraries(lob_sim PRIVATE lob_core)

add_executable(test_order_book tests/test_order_book.cpp)
target_link_libraries(test_order_book PRIVATE lob_core)

enable_testing()
add_test(NAME order_book COMMAND test_order_book)
