find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(PLENA_LIT NAMES llvm-lit lit HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED)
find_program(PLENA_FILECHECK NAMES FileCheck HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED)
configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
add_custom_target(check-plena
  COMMAND "${Python3_EXECUTABLE}" "${PLENA_LIT}" -sv "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS plena-opt plena-asm plena-compile
  USES_TERMINAL
)
add_test(NAME plena-lit
  COMMAND "${Python3_EXECUTABLE}" "${PLENA_LIT}" -sv "${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME plena-driver COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/Tools/plena-compile/driver.py"
  --compiler $<TARGET_FILE:plena-compile> --assembler $<TARGET_FILE:plena-asm>
  --root "${CMAKE_CURRENT_SOURCE_DIR}")

set(PLENA_TEST_EMULATOR "" CACHE FILEPATH "PLENA transactional emulator")
set(PLENA_TEST_SETTINGS "" CACHE FILEPATH "Simulator TOML configuration")
set(PLENA_TEST_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH "Python with PyTorch and NumPy")
if(PLENA_TEST_EMULATOR AND PLENA_TEST_SETTINGS)
  add_test(NAME plena-matmul32 COMMAND "${PLENA_TEST_PYTHON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/Integration/matmul.py"
    --compiler $<TARGET_FILE:plena-compile>
    --emulator "${PLENA_TEST_EMULATOR}" --settings "${PLENA_TEST_SETTINGS}"
    --source "${CMAKE_CURRENT_SOURCE_DIR}/Integration/matmul32.mlir"
    --graph "${CMAKE_CURRENT_SOURCE_DIR}/Integration/matmul32-graph.mlir"
            "${CMAKE_CURRENT_SOURCE_DIR}/Integration/matmul64-graph.mlir"
            "${CMAKE_CURRENT_SOURCE_DIR}/Integration/hierarchical-graph.mlir"
    --out "${CMAKE_CURRENT_BINARY_DIR}/matmul32")
  set_tests_properties(plena-matmul32 PROPERTIES TIMEOUT 300 LABELS integration)
  add_test(NAME plena-vpu32 COMMAND "${PLENA_TEST_PYTHON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/Integration/elementwise.py"
    --compiler $<TARGET_FILE:plena-compile>
    --emulator "${PLENA_TEST_EMULATOR}" --settings "${PLENA_TEST_SETTINGS}"
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
  set_tests_properties(plena-vpu32 PROPERTIES TIMEOUT 600 LABELS integration)
endif()
