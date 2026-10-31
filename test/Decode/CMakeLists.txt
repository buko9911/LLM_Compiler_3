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
add_test(NAME decode-gemv-optimizations COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/npu_gemv_optimizations.py" --optimizer $<TARGET_FILE:npu-opt>
  --output "${CMAKE_CURRENT_BINARY_DIR}/gemv-optimizations")
set_tests_properties(decode-gemv-optimizations PROPERTIES LABELS compiler)
add_test(NAME decode-projection-fusion COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/npu_projection_fusion.py" --optimizer $<TARGET_FILE:npu-opt>
  --output "${CMAKE_CURRENT_BINARY_DIR}/projection-fusion")
set_tests_properties(decode-projection-fusion PROPERTIES LABELS compiler)
add_test(NAME decode-activation-residency COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/npu_activation_residency.py" --optimizer $<TARGET_FILE:npu-opt>
  --output "${CMAKE_CURRENT_BINARY_DIR}/activation-residency")
set_tests_properties(decode-activation-residency PROPERTIES LABELS compiler)
add_test(NAME decode-attention-fusion COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/npu_attention_fusion.py" --optimizer $<TARGET_FILE:npu-opt>
  --output "${CMAKE_CURRENT_BINARY_DIR}/attention-fusion")
set_tests_properties(decode-attention-fusion PROPERTIES LABELS compiler)
add_test(NAME decode-flash-attention COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/npu_flash_attention.py" --optimizer $<TARGET_FILE:npu-opt>
  --output "${CMAKE_CURRENT_BINARY_DIR}/flash-attention")
set_tests_properties(decode-flash-attention PROPERTIES LABELS compiler)

if(NPU_TEST_EMULATOR AND NPU_TEST_SETTINGS)
  add_test(NAME decode-frontend COMMAND "${NPU_TEST_PYTHON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/decoder_frontend.py")
  set_tests_properties(decode-frontend PROPERTIES TIMEOUT 180 LABELS frontend)
endif()

add_test(NAME decode-input-pipeline COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/npu_input_pipeline.py" --opt $<TARGET_FILE:npu-opt>)
set_tests_properties(decode-input-pipeline PROPERTIES LABELS compiler)

add_test(NAME decode-matmul-layout COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/npu_matmul_layout.py" --optimizer $<TARGET_FILE:npu-opt>
  --output "${CMAKE_CURRENT_BINARY_DIR}/matmul-layout")
set_tests_properties(decode-matmul-layout PROPERTIES LABELS compiler)

add_test(NAME decode-elementwise-distribution COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/npu_elementwise_distribution.py" --optimizer $<TARGET_FILE:npu-opt>
  --output "${CMAKE_CURRENT_BINARY_DIR}/elementwise-distribution")
set_tests_properties(decode-elementwise-distribution PROPERTIES LABELS compiler)

add_test(NAME decode-prefill-layout COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/npu_prefill_layout.py" --optimizer $<TARGET_FILE:npu-opt>
  --output "${CMAKE_CURRENT_BINARY_DIR}/prefill-layout")
set_tests_properties(decode-prefill-layout PROPERTIES LABELS compiler)

add_test(NAME decode-row-distribution COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/npu_row_distribution.py" --optimizer $<TARGET_FILE:npu-opt>
  --output "${CMAKE_CURRENT_BINARY_DIR}/row-distribution")
set_tests_properties(decode-row-distribution PROPERTIES LABELS compiler)

add_test(NAME decode-runtime-backend COMMAND "${NPU_TEST_PYTHON}" -B
  "${CMAKE_CURRENT_SOURCE_DIR}/runtime_backend.py")
set_tests_properties(decode-runtime-backend PROPERTIES LABELS runtime)
