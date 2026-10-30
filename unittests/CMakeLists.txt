find_package(GTest REQUIRED)
add_executable(npu-unit Target/ISATest.cpp Target/ISAv2Test.cpp Target/IsaV2Test.cpp Target/ProgramTest.cpp Target/HardwareConfigTest.cpp
  Analysis/MemoryPlanTest.cpp Analysis/TilingPlanTest.cpp Decode/NpuZeroInitTest.cpp Decode/NpuAsyncStoreTest.cpp Decode/NpuGdmaAdmissionTest.cpp
  Decode/NpuVectorScheduleTest.cpp)
target_link_libraries(npu-unit PRIVATE NPUTarget NPUAnalysis NPUDecodeTarget MLIRParser MLIRSCFDialect
  GTest::GTest GTest::Main)
add_test(NAME npu-unit COMMAND npu-unit)
