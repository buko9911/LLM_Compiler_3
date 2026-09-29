find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(NPU_LIT NAMES llvm-lit lit HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED)
find_program(NPU_FILECHECK NAMES FileCheck HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED)
configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
add_custom_target(check-npu
  COMMAND "${Python3_EXECUTABLE}" "${NPU_LIT}" -sv "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS npu-opt npu-asm npu-compile
  USES_TERMINAL
)
add_test(NAME npu-lit
  COMMAND "${Python3_EXECUTABLE}" "${NPU_LIT}" -sv "${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME npu-driver COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/Tools/npu-compile/driver.py"
  --compiler $<TARGET_FILE:npu-compile> --assembler $<TARGET_FILE:npu-asm>
  --root "${CMAKE_CURRENT_SOURCE_DIR}")

set(NPU_TEST_EMULATOR "" CACHE FILEPATH "NPU transactional emulator")
set(NPU_TEST_SETTINGS "" CACHE FILEPATH "Simulator TOML configuration")
set(NPU_TEST_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH "Python with PyTorch and NumPy")
if(NPU_TEST_EMULATOR AND NPU_TEST_SETTINGS)
  add_test(NAME npu-matmul32 COMMAND "${NPU_TEST_PYTHON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/Integration/matmul.py"
    --compiler $<TARGET_FILE:npu-compile>
    --emulator "${NPU_TEST_EMULATOR}" --settings "${NPU_TEST_SETTINGS}"
    --source "${CMAKE_CURRENT_SOURCE_DIR}/Integration/matmul32.mlir"
    --graph "${CMAKE_CURRENT_SOURCE_DIR}/Integration/matmul32-graph.mlir"
            "${CMAKE_CURRENT_SOURCE_DIR}/Integration/matmul64-graph.mlir"
            "${CMAKE_CURRENT_SOURCE_DIR}/Integration/hierarchical-graph.mlir"
    --out "${CMAKE_CURRENT_BINARY_DIR}/matmul32")
  set_tests_properties(npu-matmul32 PROPERTIES TIMEOUT 300 LABELS integration)
  add_test(NAME npu-vpu32 COMMAND "${NPU_TEST_PYTHON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/Integration/elementwise.py"
    --compiler $<TARGET_FILE:npu-compile>
    --emulator "${NPU_TEST_EMULATOR}" --settings "${NPU_TEST_SETTINGS}"
    --source "${CMAKE_CURRENT_SOURCE_DIR}/Integration/vpu32.mlir"
    --graph "${CMAKE_CURRENT_SOURCE_DIR}/Integration/elementwise-graph.mlir"
    --softmax "${CMAKE_CURRENT_SOURCE_DIR}/Integration/softmax-graph.mlir"
    --loop "${CMAKE_CURRENT_SOURCE_DIR}/Integration/loop32.mlir"
    --attention "${CMAKE_CURRENT_SOURCE_DIR}/Integration/flashattn32.mlir"
    --carry "${CMAKE_CURRENT_SOURCE_DIR}/Integration/carry-graph.mlir"
    --flash "${CMAKE_CURRENT_SOURCE_DIR}/Integration/flashattn-graph.mlir"
    --divide "${CMAKE_CURRENT_SOURCE_DIR}/Integration/divide-graph.mlir"
    --constant "${CMAKE_CURRENT_SOURCE_DIR}/Integration/constant-graph.mlir"
    --transpose "${CMAKE_CURRENT_SOURCE_DIR}/Integration/transpose-graph.mlir"
    --layer "${CMAKE_CURRENT_SOURCE_DIR}/Integration/decoder-layer.mlir"
    --out "${CMAKE_CURRENT_BINARY_DIR}/vpu32")
  set_tests_properties(npu-vpu32 PROPERTIES TIMEOUT 600 LABELS integration)
endif()
