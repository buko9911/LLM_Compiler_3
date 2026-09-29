find_package(GTest REQUIRED)
add_executable(npu-unit Target/ISATest.cpp Target/ProgramTest.cpp Target/HardwareConfigTest.cpp
  Analysis/MemoryPlanTest.cpp Analysis/TilingPlanTest.cpp)
target_link_libraries(npu-unit PRIVATE NPUTarget NPUAnalysis GTest::GTest GTest::Main)
add_test(NAME npu-unit COMMAND npu-unit)
