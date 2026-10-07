find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(pim-layout-test pim_layout.cpp)
target_link_libraries(pim-layout-test PRIVATE NPUDecodeTarget)
add_test(NAME pim-layout COMMAND pim-layout-test)

add_test(NAME decode-driver COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/driver.py" --compiler $<TARGET_FILE:npu-compile>)
add_test(NAME decode-memplan COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/npu_memplan.py" --optimizer $<TARGET_FILE:npu-opt>
  --output "${CMAKE_CURRENT_BINARY_DIR}/memplan")
set_tests_properties(decode-memplan PROPERTIES LABELS compiler)

if(NPU_TEST_EMULATOR AND NPU_TEST_SETTINGS)
  add_test(NAME decode-frontend COMMAND "${NPU_TEST_PYTHON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/decoder_frontend.py")
  set_tests_properties(decode-frontend PROPERTIES TIMEOUT 180 LABELS frontend)
endif()
