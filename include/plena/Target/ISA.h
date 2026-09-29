#ifndef PLENA_TARGET_ISA_H
#define PLENA_TARGET_ISA_H
#include "plena/Target/Result.h"
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace plena {
// Core ISA of NPU_Simulator main, "ISA ver 1.0".
//
// Most instructions are one 32-bit word. Matrix operand loads and accumulator
// writeouts are four-word records: a header word followed by three u32
// immediates. M_MMA is one word and computes one fixed 32x32x32 tile; a long K
// is covered by issuing it once per 32-element K slice (INIT, then ACC).
//
// Numeric register indices, decimal or hexadecimal unsigned immediates.
// Results never throw; malformed operands and reserved encodings fail closed.

enum class MatrixOperand { Weight, Activation };
enum class MatrixType { F16, I8 };
enum class WriteoutType { F16, I32, F32 };

// K elements one fixed-tile M_MMA consumes. Must equal the simulator's
// [TRANSACTIONAL.MATRIX_MICROARCHITECTURE].mma_tile_k (default 32).
inline constexpr uint32_t kMmaTileK = 32;

// Weight is [K,N] (rows = K, columns = N); activation is [M,K]. Stride in bytes.
Result<std::vector<uint32_t>> encodeMatrixLoad(MatrixOperand operand, MatrixType type,
                                               unsigned addressRegister, uint32_t rows,
                                               uint32_t columns, uint32_t strideBytes);
uint32_t encodeMma(MatrixType type, bool accumulate);
Result<std::vector<uint32_t>> encodeMatrixWriteout(WriteoutType type, unsigned addressRegister,
                                                   uint32_t rows, uint32_t columns,
                                                   uint32_t strideBytes);
// The fixed-tile M_MMA sequence that covers K for one loaded operand pair.
std::vector<uint32_t> mmaSequence(MatrixType type, uint32_t k, bool accumulateFirst);

// Words occupied by the record that starts with `header` (1 or 4).
unsigned recordWords(uint32_t header);

// One instruction of source text to its record (1 or 4 words).
Result<std::vector<uint32_t>> assembleInstruction(std::string_view line);
// One-word instructions only; a matrix record is rejected.
Result<uint32_t> assembleLine(std::string_view line);
Result<std::vector<uint32_t>> assemble(std::string_view source);

// Text of the record starting at words[0]; `available` bounds payload reads.
Result<std::string> disassembleRecord(const uint32_t *words, size_t available);
// One-word instructions only.
Result<std::string> disassemble(uint32_t word);
bool isValidInstruction(uint32_t word);
// Splits a core word stream into validated records.
Result<std::vector<std::vector<uint32_t>>> splitRecords(const std::vector<uint32_t> &words);

Result<std::vector<uint32_t>> parseMem(std::string_view source);
std::string formatMem(const std::vector<uint32_t> &words);
Result<std::vector<uint32_t>> materialize(unsigned reg, uint32_t value);
struct InstructionInfo { std::string name; unsigned operands; };
std::vector<InstructionInfo> instructionSet();
} // namespace plena
#endif
