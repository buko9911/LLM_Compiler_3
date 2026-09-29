#include "npu/Target/ISA.h"
#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char **argv) {
  bool decode = false;
  std::string input = "-", output = "-";
  bool haveInput = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "usage: npu-asm [--disassemble] [input|-] [-o output|-]\n"
                   "Core ISA ver 1.0 only: numeric register indices; .mem contains one 32-bit word per line.\n"
                   "Matrix loads/writeouts are four-word records (register, rows, columns, stride_bytes);\n"
                   "M_MMA_* takes INIT or ACC.\n";
      return 0;
    }
    if (arg == "--disassemble") decode = true;
    else if (arg == "-o" && i + 1 < argc) output = argv[++i];
    else if (!haveInput && !arg.empty() && (arg == "-" || arg.front() != '-')) { input = arg; haveInput = true; }
    else { std::cerr << "invalid argument: " << arg << '\n'; return 2; }
  }
  std::ifstream file;
  if (input != "-") { file.open(input); if (!file) { std::cerr << "cannot open " << input << '\n'; return 1; } }
  std::istream &in = input == "-" ? std::cin : file;
  std::string source{std::istreambuf_iterator<char>(in), {}};
  auto words = decode ? npu::parseMem(source) : npu::assemble(source);
  if (!words) { std::cerr << words.error << '\n'; return 1; }
  std::string result;
  if (decode) {
    auto records = npu::splitRecords(words.value);
    if (!records) { std::cerr << records.error << '\n'; return 1; }
    for (const auto &r : records.value) result += npu::disassembleRecord(r.data(), r.size()).value + "\n";
  }
  else result = npu::formatMem(words.value);
  if (output == "-") { std::cout << result; return std::cout ? 0 : 1; }
  std::ofstream out(output);
  out << result;
  if (!out) { std::cerr << "cannot write " << output << '\n'; return 1; }
  return 0;
}
