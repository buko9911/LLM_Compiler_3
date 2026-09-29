add_library(NPUAnalysis MemoryPlan.cpp TilingPlan.cpp)
target_include_directories(NPUAnalysis PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(NPUAnalysis PUBLIC NPUTarget)
