find_package(GTest REQUIRED)
add_executable(npu-unit
  Support/Assembler.cpp Support/ProgramVerifier.cpp
  Target/ISATest.cpp Target/ISAv2Test.cpp
  Decode/NpuZeroInitTest.cpp Decode/NpuAsyncStoreTest.cpp Decode/NpuGdmaAdmissionTest.cpp
  Decode/NpuVectorScheduleTest.cpp)
target_include_directories(npu-unit PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(npu-unit PRIVATE NPUTarget NPUDecodeTarget MLIRParser MLIRSCFDialect
  GTest::GTest GTest::Main)
add_test(NAME npu-unit COMMAND npu-unit)
