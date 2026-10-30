find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(NPU_LIT NAMES llvm-lit lit HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED)
find_program(NPU_FILECHECK NAMES FileCheck HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED)
configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
add_custom_target(check-npu
  COMMAND "${Python3_EXECUTABLE}" "${NPU_LIT}" -sv "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS npu-opt npu-compile
  USES_TERMINAL
)
add_test(NAME npu-lit
  COMMAND "${Python3_EXECUTABLE}" "${NPU_LIT}" -sv "${CMAKE_CURRENT_BINARY_DIR}")
set(NPU_TEST_EMULATOR "" CACHE FILEPATH "NPU transactional emulator")
set(NPU_TEST_SETTINGS "" CACHE FILEPATH "Simulator TOML configuration")
set(NPU_TEST_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH "Python with PyTorch and NumPy")
