#ifndef NPU_TARGET_RESULT_H
#define NPU_TARGET_RESULT_H
#include <string>
#include <utility>

namespace npu {
template <typename T> struct Result {
  T value{};
  std::string error;
  explicit operator bool() const { return error.empty(); }
  static Result success(T value) { return {std::move(value), {}}; }
  static Result failure(std::string error) { return {{}, std::move(error)}; }
};
} // namespace npu
#endif
