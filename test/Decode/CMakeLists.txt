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

# End-to-end regressions: compile, run on the NPU emulator + Ramulator, compare with an independent reference.
# Their outputs (sparse LP6 images, SRAM dumps) can be large: point NPU_E2E_OUTPUT at a scratch disk.
set(NPU_E2E_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/e2e" CACHE PATH "Output root of the end-to-end tests")
set(NPU_TEST_LIBRARY "" CACHE FILEPATH "Ramulator bridge liblp6_ramulator.so")
function(add_e2e_test name script)
  add_test(NAME ${name} COMMAND /bin/sh -c
    "rm -rf '${NPU_E2E_OUTPUT}/${name}' && mkdir -p '${NPU_E2E_OUTPUT}' && exec \"$0\" -B \"$@\" --output '${NPU_E2E_OUTPUT}/${name}'"
    "${NPU_TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/${script}" ${ARGN})
  set_tests_properties(${name} PROPERTIES LABELS e2e TIMEOUT 900)
endfunction()
if(NPU_TEST_EMULATOR AND NPU_TEST_LIBRARY)
  set(emu --emulator "${NPU_TEST_EMULATOR}" --lp6-library "${NPU_TEST_LIBRARY}")
  set(lib --lp6-library "${NPU_TEST_LIBRARY}")
  add_e2e_test(e2e-weight-prefetch weight_prefetch.py ${lib})
  add_e2e_test(e2e-pim-scatter pim_scatter.py ${lib})
  add_e2e_test(e2e-pim-pingpong pim_pingpong.py ${emu})
  add_e2e_test(e2e-pim-weight-storage pim_weight_storage.py ${emu})
  add_e2e_test(e2e-pim-packet-npu pim_packet_npu.py ${lib})
  add_e2e_test(e2e-pim-activation-cores pim_activation_cores.py --compiler $<TARGET_FILE:npu-compile>
    --baseline "${NPU_E2E_OUTPUT}/e2e-pim-packet-npu" ${lib})
  set_tests_properties(e2e-pim-activation-cores PROPERTIES DEPENDS e2e-pim-packet-npu)
  add_e2e_test(e2e-pim-activation-ranks pim_activation_ranks.py --compiler $<TARGET_FILE:npu-compile>
    --emulator "${NPU_TEST_EMULATOR}" --library "${NPU_TEST_LIBRARY}")
  add_e2e_test(e2e-l2-dependencies npu_l2_dependencies.py
    "${NPU_E2E_OUTPUT}/e2e-weight-prefetch/auto_ring/pipeline" --negative-control)
  set_tests_properties(e2e-l2-dependencies PROPERTIES DEPENDS e2e-weight-prefetch)
  add_e2e_test(e2e-vector-schedule npu_vector_schedule_runtime.py)
  add_e2e_test(e2e-gemv-optimizations npu_gemv_optimizations_runtime.py)
  add_e2e_test(e2e-attention-fusion npu_attention_fusion_runtime.py)
  add_e2e_test(e2e-dependency-schedule npu_dependency_schedule_runtime.py)
  add_e2e_test(e2e-elementwise-distribution npu_elementwise_distribution_runtime.py)
  add_e2e_test(e2e-flash-attention npu_flash_attention_runtime.py)
  add_e2e_test(e2e-matmul-layout npu_matmul_layout_runtime.py)
  add_e2e_test(e2e-projection-fusion npu_projection_fusion_runtime.py)
  add_test(NAME decode-target-simulators COMMAND "${NPU_TEST_PYTHON}" -B
    "${CMAKE_CURRENT_SOURCE_DIR}/target_simulators.py")
  set_tests_properties(decode-target-simulators PROPERTIES LABELS compiler)
endif()
