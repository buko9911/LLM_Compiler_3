find_package(GTest REQUIRED)
add_executable(plena-unit Target/ISATest.cpp Target/ProgramTest.cpp Target/HardwareConfigTest.cpp
  Analysis/MemoryPlanTest.cpp Analysis/TilingPlanTest.cpp)
target_link_libraries(plena-unit PRIVATE PlenaTarget PlenaAnalysis GTest::GTest GTest::Main)
add_test(NAME plena-unit COMMAND plena-unit)
