add_library(PlenaAnalysis MemoryPlan.cpp TilingPlan.cpp)
target_include_directories(PlenaAnalysis PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(PlenaAnalysis PUBLIC PlenaTarget)
