//===----------------------------------------------------------------------===//
// npu-compile --pipeline=graph-hybrid — NPU + LPDDR6-PIM 하이브리드 컴파일러 드라이버
//
// linalg 그래프 하나(Stage 0 캡처)를 W8A8 로 합법화한 뒤, 가중치 GEMV 묶음마다
// NPU 와 PIM 중 어디서 돌릴지 비용 모델로 정하고(partitionGraph), PIM 묶음을
// 경계로 NPU 페이즈들을 잘라(outlinePhases) 페이즈마다 npu-compile 과 같은
// tile / place / lower / encode 를 돌린다. PIM 묶음은 PIM primitive 명령이 된다
// (lib/Target/PimLayout.cpp).
//
//   NPU 0 | PIM 0 | NPU 1 | PIM 1 | ... | PIM G-1 | NPU G
//
// 출력 디렉터리:
//   program.bin, system.json   디코드 한 스텝 = Unified Program 하나 (ISA ver 2.0).
//                              NPU 페이즈는 레코드 구간, PIM 페이즈는 PIM 레코드 하나
//                              (PIM primitive 명령들). system.json 에 PIM 커널 설정과
//                              레코드 이름 <페이즈>.<종류>_<번호> (npu<i>, pim<g>).
//   partition.json             묶음별 비용 추정과 선택
//   metadata.json              전역 주소, 출력의 L2 readback 주소, PIM 요약
//   lp6.bin                    LP6 이미지 (--inputs 가 있을 때)
//   phase<i>/                  --save-stages 일 때 페이즈 i 의 단계별 MLIR
//
// 페이즈 순서는 레코드 의존성이다(linkProgram). NPU 페이즈의 의존성 없는
// 레코드는 앞 페이즈의 끝을 기다리고, 페이즈 끝은 BARRIER 하나다.
//
// 페이즈 경계는 GDMA 명령으로 이어 붙인다. NPU 는 PIM 을 모른다.
//   * PIM 묶음 g 의 INT8 활성값은 NPU 페이즈 g 끝에서 DRAM 전역 -> L2 -> PIM
//     활성값 행(읽는 파티션의 PU 8개 뱅크마다)으로 뿌린다.
//   * PIM 묶음 g 의 INT32 결과는 NPU 페이즈 g+1 시작에서 PIM 출력 행 -> L2 ->
//     DRAM 전역으로 모은다. 그 뒤로는 보통 입력처럼 읽힌다.
//   * 마지막 NPU 페이즈 끝에서 출력들을 L2 0 번지부터 읽어 둔다. 시뮬레이터는
//     DRAM 이 아니라 L2 를 덤프한다.
//
// 종료 코드: 0 성공, 1 컴파일 실패, 2 사용법 오류.
//===----------------------------------------------------------------------===//
#include "npu/Dialect/NPUDialect.h"
#include "npu/Target/HardwareConfig.h"
#include "npu/Target/IsaV2.h"
#include "npu/Target/PimLayout.h"
#include "npu/Target/Program.h"
#include "npu/Target/Quantize.h"
#include "npu/Transforms/Hybrid.h"
#include "npu/Transforms/Passes.h"
#include "npu/Transforms/Pipeline.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace {
namespace fs = std::filesystem;
namespace pim = ::npu::pim;
using namespace mlir;

struct Options {
  std::string input, output, settings, inputsDir, config, names, policy;
  std::vector<std::pair<std::string, std::string>> overrides;
  uint64_t lp6Bytes = uint64_t(16) << 30;
  int64_t splitEvery = 0;
  bool saveStages = false, timeStages = false;
};

uint64_t alignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

bool save(ModuleOp module, const fs::path &path) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path.string(), ec);
  if (ec) return false;
  module.print(out);
  out << '\n';
  return !out.has_error();
}

bool writeJSON(const fs::path &path, llvm::json::Value value) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path.string(), ec);
  if (ec) return false;
  out << llvm::formatv("{0:1}", value) << '\n';
  return !out.has_error();
}

// arguments.txt of a capture: "<index> <name> <shape> <dtype>" per line.
std::vector<std::string> readNames(const fs::path &path) {
  std::vector<std::string> names;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream fields(line);
    int64_t index;
    std::string name;
    if (!(fields >> index >> name) || index < 0) continue;
    if (names.size() <= size_t(index)) names.resize(size_t(index) + 1);
    names[size_t(index)] = name;
  }
  return names;
}

std::string dtypeOf(Type type) {
  auto element = cast<ShapedType>(type).getElementType();
  if (element.isF16()) return "float16";
  if (element.isF32()) return "float32";
  if (element.isInteger(8)) return "int8";
  if (element.isInteger(32)) return "int32";
  return "unknown";
}

// One GDMA command of a phase-boundary transfer.
struct Dma {
  bool load = true;
  uint64_t dram = 0, l2 = 0, bytes = 0, rows = 1, dramStride = 0, l2Stride = 0;
};
void emitDma(OpBuilder &b, Location loc, const Dma &d) {
  OperationState state(loc, "npu.dma");
  state.addAttribute("direction", b.getStringAttr(d.load ? "load" : "store"));
  state.addAttribute("dram", b.getI64IntegerAttr(int64_t(d.dram)));
  state.addAttribute("l2", b.getI64IntegerAttr(int64_t(d.l2)));
  state.addAttribute("bytes", b.getI64IntegerAttr(int64_t(d.bytes)));
  if (d.rows > 1) {
    state.addAttribute("rows", b.getI64IntegerAttr(int64_t(d.rows)));
    state.addAttribute("dram_stride", b.getI64IntegerAttr(int64_t(d.dramStride)));
    state.addAttribute("l2_stride", b.getI64IntegerAttr(int64_t(d.l2Stride)));
  }
  b.create(state);
}

// The INT8 activation [M, K] of a PIM group into the activation rows.
//
// Under rarobacobgch the column is above the bank group and the subchannel, so
// one column of one bank is a block of 4 bank groups x C subchannels x 32 B
// (colBytes: 1 KiB for four LPDDR6 channels, 256 B for one). The even PUs
// 0,2,4,6 sit in bank 0 of bank groups 0..3 and the odd PUs in bank 2, and
// every PU reads the same packet from its own bank. So packet i, copied 4 C
// times into a block, is what column i of bank 0 and of bank 2 hold in every
// rank the group computes on. 4 C GDMA loads build those blocks in L2 (the
// packet 32 B apart in DRAM, one block apart in L2), and then one store per bank
// and DRAM row writes n whole columns as n contiguous blocks. The slots of a
// subchannel the group does not use belong to this activation too (its tile id
// owns that column in every partition), so writing them is free.
std::vector<Dma> scatter(const pim::Program &program, const ::npu::PimGroup &group,
                         const std::vector<int64_t> &activations, const ::npu::HybridGlobal &input,
                         uint64_t staging, uint64_t stagingBytes, std::string &error) {
  std::vector<Dma> out;
  const int64_t block = program.colBytes(), perChunk = int64_t(stagingBytes) / block;
  const int slots = 4 * program.channels;
  if (!perChunk) {
    error = "L2 staging window smaller than one column block";
    return out;
  }
  for (int64_t m = 0; m < group.tokens; ++m) {
    const pim::Activation &a = program.activations[size_t(activations[size_t(m)])];
    std::set<int> ranks;
    for (int part : a.parts) ranks.insert(program.partRank(part));
    for (int64_t chunk = 0; chunk < a.packets(); chunk += perChunk) {
      const int64_t count = std::min(perChunk, a.packets() - chunk);
      const uint64_t source = input.address + uint64_t(m * group.k + chunk * 32);
      for (int slot = 0; slot < slots; ++slot)   // bank group slot / C, subchannel slot % C
        out.push_back({true, source, staging + uint64_t(slot) * 32, 32, uint64_t(count), 32, uint64_t(block)});
      for (int rank : ranks)
        for (int parity = 0; parity < 2; ++parity)   // PU 0 -> bank 0 of group 0, PU 1 -> bank 2 of group 0
          for (int64_t first = chunk; first < chunk + count;) {
            int64_t packet = a.packetBase() + first;
            int64_t n = std::min(chunk + count - first, pim::kPacketsPerRow - packet % pim::kPacketsPerRow);
            out.push_back({false, program.activationPacketAddress(a, first, parity, rank * program.channels),
                           staging + uint64_t((first - chunk) * block), uint64_t(n * block)});
            first += n;
          }
    }
  }
  return out;
}

// The INT32 results [M, N] of one PIM GEMV: stored tiles (8 outputs of one PU
// each) -> L2 in output order, chunk by chunk, then L2 -> the DRAM global. A
// run of tiles stops at a DRAM row and where the stream's partition changes.
std::vector<Dma> gather(const pim::Program &program, const pim::Projection &p, const ::npu::HybridGlobal &result,
                        int64_t tokens, uint64_t staging, uint64_t stagingBytes) {
  std::vector<Dma> out;
  const int64_t N = p.n, rows = p.rowsPerPU(), tiles = p.tilesPerPU();
  const int64_t chunk = int64_t(stagingBytes / 4) / 8 * 8;
  auto partitionEnd = [&](int64_t t) {
    int64_t g = t / 8;
    while (g < p.streams() && p.streamPart[size_t(g)] == p.streamPart[size_t(t / 8)]) ++g;
    return g * 8;
  };
  for (int m = 0; m < int(tokens); ++m)
    for (int64_t c0 = 0; c0 < N; c0 += chunk) {
      int64_t c1 = std::min(N, c0 + chunk);
      for (int pu = 0; pu < pim::kPUs; ++pu) {
        int64_t low = std::max(c0, pu * rows), high = std::min(c1, (pu + 1) * rows);
        if (low >= high) continue;
        int64_t first = (low - pu * rows) / 8, last = (high - pu * rows + 7) / 8;
        for (int64_t t = first; t < last && t < tiles;) {
          int64_t tile = p.tokenOutputTile(m, t), packet = p.outputBase + tile;
          int64_t n = std::min({last - t, pim::kPacketsPerRow - packet % pim::kPacketsPerRow, partitionEnd(t) - t});
          out.push_back({true, program.outputTileAddress(p, tile, pu),
                         staging + uint64_t((pu * rows + t * 8 - c0) * 4), 32, uint64_t(n),
                         uint64_t(program.colBytes()), 32});
          t += n;
        }
      }
      out.push_back({false, result.address + uint64_t(m * N * 4 + c0 * 4), staging, uint64_t((c1 - c0) * 4)});
    }
  return out;
}

// A sparse LP6 image mapped into memory: writes are stores, untouched pages stay holes.
class Image {
  int fd = -1;
  uint8_t *base = nullptr;
  uint64_t size = 0;
public:
  bool open(const fs::path &path, uint64_t bytes) {
    size = bytes;
    fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 || ::ftruncate(fd, off_t(bytes))) return false;
    void *mapped = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) return false;
    base = static_cast<uint8_t *>(mapped);
    return true;
  }
  bool write(uint64_t address, const void *data, uint64_t bytes) {
    if (address > size || bytes > size - address) return false;
    std::memcpy(base + address, data, bytes);
    return true;
  }
  ~Image() {
    if (base) ::munmap(base, size);
    if (fd >= 0) ::close(fd);
  }
};

bool readFile(const fs::path &path, std::vector<char> &bytes) {
  std::error_code ec;
  auto size = fs::file_size(path, ec);
  if (ec) return false;
  bytes.resize(size);
  std::ifstream in(path, std::ios::binary);
  in.read(bytes.data(), std::streamsize(size));
  return bool(in);
}
} // namespace

int compileHybridGraph(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto value = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
    if (arg == "--help") {
      std::cout << "usage: npu-compile --pipeline=graph-hybrid graph.mlir --settings hardware.toml -o directory\n"
                   "  --inputs dir          arg<N>.bin of the capture; builds lp6.bin\n"
                   "  --config hybrid.json  cost model and mapping policy (default: built-in values)\n"
                   "  --policy cost|npu|pim override the configured policy\n"
                   "  --map name=npu|pim    map every group whose weight name contains name (repeatable)\n"
                   "  --names file          argument names (default: <inputs>/arguments.txt)\n"
                   "  --split-every N       also start an NPU phase at every N-th GEMV group (4 = per Qwen3 layer)\n"
                   "  --lp6-bytes N         LP6 capacity (default 16 GiB)\n"
                   "  --save-stages | --time-stages\n";
      return 0;
    }
    if (arg == "-o") options.output = value();
    else if (arg == "--settings") options.settings = value();
    else if (arg == "--inputs") options.inputsDir = value();
    else if (arg == "--config") options.config = value();
    else if (arg == "--names") options.names = value();
    else if (arg == "--policy") options.policy = value();
    else if (arg == "--map") {
      std::string map = value();
      auto eq = map.find('=');
      if (eq == std::string::npos) { std::cerr << "--map takes name=npu|pim\n"; return 2; }
      options.overrides.push_back({map.substr(0, eq), map.substr(eq + 1)});
    } else if (arg == "--lp6-bytes") options.lp6Bytes = std::stoull(value());
    else if (arg == "--split-every") options.splitEvery = std::stoll(value());
    else if (arg == "--save-stages") options.saveStages = true;
    else if (arg == "--time-stages") options.timeStages = true;
    else if (options.input.empty() && !arg.empty() && arg[0] != '-') options.input = arg;
    else { std::cerr << "invalid argument: " << arg << '\n'; return 2; }
  }
  if (options.input.empty() || options.output.empty() || options.settings.empty()) {
    std::cerr << "requires graph.mlir --settings hardware.toml -o directory\n";
    return 2;
  }
  auto clock = std::chrono::steady_clock::now();
  auto timed = [&](const char *what) {
    auto now = std::chrono::steady_clock::now();
    if (options.timeStages)
      std::cerr << "timing " << what << ' ' << std::chrono::duration<double>(now - clock).count() << " s\n";
    clock = now;
  };

  DialectRegistry registry;
  ::npu::registerPipelineDialects(registry);
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = parseSourceFile<ModuleOp>(options.input, &context);
  if (!module || failed(verify(*module))) return 1;
  auto hardware = ::npu::loadHardwareConfig(options.settings);
  if (!hardware) { std::cerr << hardware.error << '\n'; return 1; }
  const auto &hw = hardware.value;
  auto model = ::npu::HybridCostModel::load(options.config);
  if (!model) { std::cerr << model.error << '\n'; return 1; }
  if (!options.policy.empty()) model.value.policy = options.policy;
  for (auto &o : options.overrides) model.value.overrides.push_back(o);
  if (model.value.policy != "cost" && model.value.policy != "npu" && model.value.policy != "pim") {
    std::cerr << "policy must be cost, npu or pim\n";
    return 2;
  }
  fs::path out(options.output);
  std::error_code ec;
  fs::create_directories(out, ec);
  if (ec) { std::cerr << ec.message() << '\n'; return 1; }
  // 예전 형식(페이즈별 program.bin, pim.bin, schedule.json)의 남은 출력.
  for (const char *stale : {"pim.bin", "schedule.json", "phases.json"}) fs::remove(out / stale, ec);
  for (const auto &entry : fs::directory_iterator(out, ec)) {
    const std::string name = entry.path().filename().string();
    if (entry.is_directory() && name.rfind("phase", 0) == 0 && name.size() > 5 &&
        name.find_first_not_of("0123456789", 5) == std::string::npos)
      fs::remove_all(entry.path(), ec);
  }
  std::vector<std::string> names = readNames(!options.names.empty() ? fs::path(options.names)
                                             : !options.inputsDir.empty() ? fs::path(options.inputsDir) / "arguments.txt"
                                                                          : fs::path());

  //===--------------------------------------------------------------------===//
  // Stage 1: 합법화(정규화 + W8A8 + 검사 + 상수 올리기)
  //===--------------------------------------------------------------------===//
  ::npu::attachHardware(*module, hw);
  (*module)->setAttr("npu.stage", StringAttr::get(&context, "graph"));
  (*module)->setAttr("npu.numerical", StringAttr::get(&context, "fp16"));
  (*module)->setAttr("npu.division", StringAttr::get(&context, "reciprocal"));
  (*module)->setAttr("npu.quantize", StringAttr::get(&context, "w8a8"));
  if (failed(::npu::legalizeGraph(*module)) || failed(verify(*module))) return 1;
  if (options.saveStages && !save(*module, out / "01-legal.mlir")) return 1;
  timed("legalize");

  // 이미지를 만들 때 필요한 것들: 전치를 접은 인자, 올린 상수, 양자화 출처.
  auto entry = *module->getOps<func::FuncOp>().begin();
  std::set<int64_t> pretransposed;
  if (auto flipped = (*module)->getAttrOfType<DenseI64ArrayAttr>("npu.pretransposed"))
    for (auto index : flipped.asArrayRef()) pretransposed.insert(index);
  std::map<int64_t, std::vector<char>> constants;
  if (auto held = (*module)->getAttrOfType<ArrayAttr>("npu.constants")) {
    unsigned first = entry.getNumArguments() - unsigned(held.size());
    for (auto [i, value] : llvm::enumerate(held)) {
      std::vector<char> raw;
      for (auto element : cast<DenseElementsAttr>(value).getValues<APFloat>()) {
        auto bits = element.bitcastToAPInt();
        for (unsigned byte = 0; byte < bits.getBitWidth() / 8; ++byte)
          raw.push_back(char(bits.extractBitsAsZExtValue(8, byte * 8)));
      }
      constants[first + int64_t(i)] = std::move(raw);
    }
  }
  struct Origin { int64_t source = -1; std::string part; };
  std::map<int64_t, Origin> quantized;
  for (unsigned i = 0; i < entry.getNumArguments(); ++i)
    if (auto origin = entry.getArgAttrOfType<DictionaryAttr>(i, "npu.quantized"))
      quantized[i] = {origin.getAs<IntegerAttr>("source").getInt(), origin.getAs<StringAttr>("part").getValue().str()};

  //===--------------------------------------------------------------------===//
  // 분할과 페이즈 나누기
  //===--------------------------------------------------------------------===//
  auto report = ::npu::partitionGraph(*module, model.value, names);
  if (failed(report)) return 1;
  {
    std::ofstream file(out / "partition.json");
    file << *report << '\n';
  }
  auto plan = ::npu::outlinePhases(*module, names, options.splitEvery);
  if (failed(plan)) return 1;
  auto &globals = plan->globals;
  const int64_t groups = int64_t(plan->pimGroups.size());
  timed("partition");

  // 전역 DRAM: 인자, 페이즈를 건너는 값, PIM 결과를 차례로. 페이즈 안에서만
  // 사는 버퍼는 그 위(npu.dram_base)에 페이즈마다 따로 놓인다.
  uint64_t cursor = 0;
  for (auto &g : globals) {
    cursor = alignUp(cursor, 256);
    g.address = cursor;
    cursor += g.bytes;
  }
  const uint64_t scratchBase = alignUp(cursor, 1 << 16);

  //===--------------------------------------------------------------------===//
  // NPU 페이즈: tile, place
  //===--------------------------------------------------------------------===//
  auto runPass = [&](ModuleOp m, std::unique_ptr<Pass> pass) {
    PassManager pm(&context);
    pm.addPass(std::move(pass));
    return succeeded(pm.run(m));
  };
  auto integer = [&](int64_t v) { return IntegerAttr::get(IntegerType::get(&context, 64), v); };
  const int64_t phases = int64_t(plan->npuPhases.size());
  std::vector<bool> empty(static_cast<size_t>(phases), false);
  uint64_t dramEnd = scratchBase;
  for (int64_t p = 0; p < phases; ++p) {
    ModuleOp m = *plan->npuPhases[size_t(p)];
    auto f = *m.getOps<func::FuncOp>().begin();
    fs::path dir = out / ("phase" + std::to_string(p));
    if (options.saveStages) fs::create_directories(dir, ec);
    for (unsigned i = 0; i < f.getNumArguments(); ++i) {
      auto id = f.getArgAttrOfType<IntegerAttr>(i, "npu.global").getInt();
      f.setArgAttr(i, "npu.address", integer(int64_t(globals[size_t(id)].address)));
    }
    for (unsigned i = 0; i < f.getNumResults(); ++i) {
      auto id = f.getResultAttrOfType<IntegerAttr>(i, "npu.global").getInt();
      f.setResultAttr(i, "npu.address", integer(int64_t(globals[size_t(id)].address)));
    }
    m->setAttr("npu.dram_base", integer(int64_t(scratchBase)));
    if (options.saveStages && !save(m, dir / "01-legal.mlir")) return 1;
    if (llvm::hasSingleElement(f.front())) {   // 반환 말고는 아무것도 없다
      empty[size_t(p)] = true;
      continue;
    }
    if (!runPass(m, ::npu::createTilePass(hw))) { std::cerr << "phase " << p << ": tiling failed\n"; return 1; }
    if (options.saveStages && !save(m, dir / "02-tiled.mlir")) return 1;
    if (!runPass(m, ::npu::createPlacePass(hw))) { std::cerr << "phase " << p << ": placement failed\n"; return 1; }
    if (options.saveStages && !save(m, dir / "03-placed.mlir")) return 1;
    dramEnd = std::max<uint64_t>(dramEnd, uint64_t(m->getAttrOfType<IntegerAttr>("npu.dram_bytes").getInt()));
    // 이 페이즈의 행렬곱이 가중치를 패널 단위로 모아 달라고 했으면 이미지가 알아야 한다.
    f = *m.getOps<func::FuncOp>().begin();
    for (unsigned i = 0; i < f.getNumArguments(); ++i)
      if (auto panel = f.getArgAttrOfType<IntegerAttr>(i, "npu.packed_panel")) {
        auto &g = globals[size_t(f.getArgAttrOfType<IntegerAttr>(i, "npu.global").getInt())];
        if (g.packedPanel && g.packedPanel != panel.getInt()) {
          std::cerr << g.name << " is packed differently by two phases\n";
          return 1;
        }
        g.packedPanel = panel.getInt();
      }
  }
  timed("tile+place");

  //===--------------------------------------------------------------------===//
  // PIM: 타일 배치와 PIM primitive 명령
  //===--------------------------------------------------------------------===//
  pim::Program program;
  program.channels = model.value.pimChannels;
  program.partitions = model.value.pimPartitions;
  if (program.channels < 1 || program.partitions < 1 || program.partitions > program.channels * pim::kRanks) {
    std::cerr << "pim.partitions must be 1 .. pim.channels x " << pim::kRanks << '\n';
    return 1;
  }
  program.nextWeight.assign(size_t(program.partitions), 0);
  program.kernelRow = int64_t(alignUp(dramEnd, uint64_t(program.rowBytes())) / uint64_t(program.rowBytes())) + 1;
  std::vector<std::vector<int64_t>> activationsOf, projectionsOf;
  activationsOf.resize(static_cast<size_t>(groups));
  projectionsOf.resize(static_cast<size_t>(groups));
  std::vector<int> rankOf(static_cast<size_t>(groups), -1);
  for (int64_t g = 0; g < groups; ++g) {
    const auto &group = plan->pimGroups[size_t(g)];
    if (group.k % pim::kKPerSweep) { std::cerr << "PIM group " << g << ": K % 256 != 0\n"; return 1; }
    for (int64_t m = 0; m < group.tokens; ++m) activationsOf[size_t(g)].push_back(program.addActivation(group.k));
    std::map<int, int64_t> loads;
    if (!model.value.pimConcurrentRanks && program.partitions > program.channels) {
      // 한 페이즈는 한 rank 의 두 서브채널에서 돈다. 가중치를 덜 가진 rank 로.
      int rank = 0;
      for (int r = 1; r <= program.partRank(program.partitions - 1); ++r)
        if (program.storedTiles(r) < program.storedTiles(rank)) rank = r;
      for (int c = 0; c < program.partitions; ++c)
        if (program.partRank(c) != rank) loads[c] = std::numeric_limits<int64_t>::max() / 4;
      rankOf[size_t(g)] = rank;
    }
    for (const auto &gemv : group.gemvs) {
      if (gemv.k != group.k) { std::cerr << "PIM group " << g << ": GEMVs read different K\n"; return 1; }
      if (gemv.n % 8) { std::cerr << gemv.name << ": N % 8 != 0\n"; return 1; }
      projectionsOf[size_t(g)].push_back(
          program.addProjection(gemv.name, gemv.n, gemv.k, activationsOf[size_t(g)], loads));
    }
  }
  // PIM 페이즈 g 의 명령 범위(first, count). 최종 primitive 단위.
  std::vector<pim::Primitive> pimCommands;
  std::vector<std::pair<int64_t, int64_t>> pimPhases;
  for (int64_t g = 0; g < groups; ++g) {
    int64_t first = int64_t(pimCommands.size());
    for (int64_t index : projectionsOf[size_t(g)])
      if (auto emitted = program.emitCompute(index, pimCommands); !emitted) {
        std::cerr << emitted.error << '\n';
        return 1;
      }
    pimPhases.push_back({first, int64_t(pimCommands.size()) - first});
  }
  const int64_t pimCommandCount = int64_t(pimCommands.size());
  {
    std::ofstream ir(out / "pim_primitives.txt");
    if (!ir) { std::cerr << "cannot write PIM primitive IR\n"; return 1; }
    ir << pim::printProgram(pimCommands);
  }
  if (program.rankEndAddress() > uint64_t(pim::kRankBytes)) {
    std::cerr << "PIM rows end past one rank\n";
    return 1;
  }
  const uint64_t imageBytes = alignUp(std::max(dramEnd, program.endAddress()), 1 << 16);
  if (imageBytes > options.lp6Bytes) {
    std::cerr << "LP6 plan needs " << imageBytes << " bytes, capacity " << options.lp6Bytes << '\n';
    return 1;
  }
  timed("pim-plan");

  //===--------------------------------------------------------------------===//
  // NPU 페이즈: lower, 경계 전송, encode
  //===--------------------------------------------------------------------===//
  llvm::json::Array outputs;
  llvm::json::Array phaseInfo;
  std::vector<::npu::Segment> segments;
  int64_t l2Required = 0, cores = 1;
  for (int64_t p = 0; p < phases; ++p) {
    ModuleOp m = *plan->npuPhases[size_t(p)];
    fs::path dir = out / ("phase" + std::to_string(p));
    uint64_t staging = 0, stagingBytes = ::npu::kDRAMStagingBytes;
    if (!empty[size_t(p)]) {
      if (!runPass(m, ::npu::createLowerPass())) { std::cerr << "phase " << p << ": lowering failed\n"; return 1; }
      staging = uint64_t(m->getAttrOfType<IntegerAttr>("npu.l2_staging").getInt());
      stagingBytes = uint64_t(m->getAttrOfType<IntegerAttr>("npu.l2_staging_bytes").getInt());
      if (auto metadata = m->getAttrOfType<StringAttr>("npu.metadata"); metadata && options.saveStages) {
        std::ofstream file(dir / "metadata.json");
        file << metadata.getValue().str();
      }
    } else {
      // 할 일이 전송뿐인 페이즈. 모듈 본문을 비우고 용량만 적는다.
      auto f = *m.getOps<func::FuncOp>().begin();
      f.erase();
      m->setAttr("npu.l1_bytes", integer(64));
      m->setAttr("npu.l2_bytes", integer(int64_t(stagingBytes)));
    }
    OpBuilder b(&context);
    uint64_t l2End = uint64_t(m->getAttrOfType<IntegerAttr>("npu.l2_bytes").getInt());
    // 앞에: 직전 PIM 묶음의 결과를 DRAM 전역으로.
    if (int64_t g = p > 0 ? plan->pimAfter[size_t(p - 1)] : -1; g >= 0) {
      const auto &group = plan->pimGroups[size_t(g)];
      b.setInsertionPointToStart(m.getBody());
      for (auto [j, gemv] : llvm::enumerate(group.gemvs))
        for (const auto &d : gather(program, program.projections[size_t(projectionsOf[size_t(g)][j])],
                                    globals[size_t(gemv.result)], group.tokens, staging, stagingBytes))
          emitDma(b, m.getLoc(), d);
      l2End = std::max(l2End, staging + stagingBytes);
    }
    b.setInsertionPointToEnd(m.getBody());
    // 뒤에: 다음 PIM 묶음의 활성값을 PIM 행으로.
    if (int64_t g = p + 1 < phases ? plan->pimAfter[size_t(p)] : -1; g >= 0) {
      const auto &group = plan->pimGroups[size_t(g)];
      std::string error;
      auto dmas = scatter(program, group, activationsOf[size_t(g)], globals[size_t(group.input)], staging,
                          stagingBytes, error);
      if (!error.empty()) { std::cerr << "phase " << p << ": " << error << '\n'; return 1; }
      for (const auto &d : dmas) emitDma(b, m.getLoc(), d);
      l2End = std::max(l2End, staging + stagingBytes);
    }
    // 마지막 페이즈 끝에: 출력을 L2 0 번지부터.
    if (p == phases - 1) {
      uint64_t at = 0;
      for (int64_t id : plan->outputs) {
        const auto &g = globals[size_t(id)];
        emitDma(b, m.getLoc(), {true, g.address, at, g.bytes});
        auto shape = cast<RankedTensorType>(g.type).getShape();
        outputs.push_back(llvm::json::Object{{"name", g.name},
                                             {"global", id},
                                             {"address", int64_t(g.address)},
                                             {"bytes", int64_t(g.bytes)},
                                             {"dtype", dtypeOf(g.type)},
                                             {"shape", llvm::json::Array(shape)},
                                             {"readback_l2_address", int64_t(at)}});
        at = alignUp(at + g.bytes, 64);
      }
      l2End = std::max(l2End, at);
    }
    // PIM 행은 DRAM 전역보다 훨씬 위에 있다. 주소 검사가 LP6 전체를 보게 한다.
    m->setAttr("npu.dram_bytes", integer(int64_t(options.lp6Bytes)));
    m->setAttr("npu.l2_bytes", integer(int64_t(std::min<uint64_t>(alignUp(l2End, 64), hw.l2Bytes))));
    if (options.saveStages && !save(m, dir / "03-target.mlir")) return 1;
    if (!runPass(m, ::npu::createEncodePass())) { std::cerr << "phase " << p << ": encoding failed\n"; return 1; }
    auto encoded = *m.getOps<mlir::npu::ProgramOp>().begin();
    ::npu::Segment segment{"npu" + std::to_string(p), {}, {}};
    for (int32_t word : encoded.getWords()) segment.program.push_back(uint32_t(word));
    // 페이즈 매니페스트에서 L2 요구량과 코어 수만 가져온다.
    auto manifest = llvm::json::parse(encoded.getManifest().str());
    if (!manifest) { llvm::consumeError(manifest.takeError()); std::cerr << "phase " << p << ": bad manifest\n"; return 1; }
    if (auto *o = manifest->getAsObject()) {
      l2Required = std::max(l2Required, o->getInteger("l2_bytes_required").value_or(0));
      if (auto *placement = o->getArray("logical_to_physical")) cores = std::max(cores, int64_t(placement->size()));
    }
    phaseInfo.push_back(llvm::json::Object{{"phase", p},
                                           {"name", segment.name},
                                           {"records", int64_t(segment.program[3])},
                                           {"transfer_only", bool(empty[size_t(p)])}});
    segments.push_back(std::move(segment));
    if (int64_t g = p + 1 < phases ? plan->pimAfter[size_t(p)] : -1; g >= 0) {
      auto [first, count] = pimPhases[size_t(g)];
      segments.push_back({"pim" + std::to_string(g), {},
                          std::vector<pim::Primitive>(pimCommands.begin() + first,
                                                  pimCommands.begin() + first + count)});
    }
  }
  timed("lower+encode");

  // 페이즈들을 프로그램 하나로.
  auto linked = ::npu::linkProgram(segments);
  if (!linked) { std::cerr << "program.bin: " << linked.error << '\n'; return 1; }
  const auto &words = linked.value.words;
  {
    std::ofstream binary(out / "program.bin", std::ios::binary);
    for (uint32_t word : words)   // little-endian
      for (unsigned byte = 0; byte < 4; ++byte) binary.put(char(word >> (8 * byte)));
    if (!binary) return 1;
  }
  llvm::json::Array symbols, placement;
  for (auto [id, name] : llvm::enumerate(linked.value.symbols))
    symbols.push_back(llvm::json::Object{{"id", int64_t(id)}, {"name", name}});
  for (int64_t c = 0; c < cores; ++c) placement.push_back(c);
  llvm::json::Object system{{"schema", ::npu::kProgramSchema},
                            {"program", "program.bin"},
                            {"scheduling_policy", "compiler_static"},
                            {"logical_to_physical", std::move(placement)},
                            {"program_word_count", int64_t(words[2])},
                            {"command_count", int64_t(words[3])},
                            {"core_instruction_count", int64_t(words[4])},
                            {"command_symbols", std::move(symbols)},
                            {"l2_bytes_required", l2Required},
                            {"l2_regions", llvm::json::Array{llvm::json::Object{
                                {"name", "l2"}, {"offset", 0}, {"size", l2Required}, {"alignment", 64}}}}};
  if (groups) system["pim_primitive_abi"] = int64_t(pim::kPrimitiveVersion);
  if (groups)
    system["pim_kernel"] = llvm::json::Object{{"channel", 0}, {"rank", 0}, {"row", program.kernelRow},
                                              {"start_column", 0}, {"bank_set", 1}, {"weight_bits", 8},
                                              {"activation_bits", 8}, {"scale_bits", 0},
                                              {"input_register_count", 8}, {"output_register_count", 8},
                                              {"subchannels", program.channels}};
  if (!writeJSON(out / "system.json", std::move(system))) return 1;

  llvm::json::Array globalInfo, pimInfo;
  for (auto [id, g] : llvm::enumerate(globals)) {
    const char *kind = g.kind == ::npu::HybridGlobal::Kind::Argument ? "argument"
                     : g.kind == ::npu::HybridGlobal::Kind::PimOutput ? "pim_output" : "value";
    llvm::json::Object entryInfo{{"id", int64_t(id)}, {"name", g.name}, {"kind", kind},
                                 {"address", int64_t(g.address)}, {"bytes", int64_t(g.bytes)},
                                 {"dtype", dtypeOf(g.type)},
                                 {"shape", llvm::json::Array(cast<RankedTensorType>(g.type).getShape())}};
    if (g.argument >= 0) entryInfo["argument"] = g.argument;
    if (g.packedPanel) entryInfo["packed"] = g.packedPanel;
    globalInfo.push_back(std::move(entryInfo));
  }
  for (int64_t g = 0; g < groups; ++g) {
    llvm::json::Array gemvs;
    for (auto [j, gemv] : llvm::enumerate(plan->pimGroups[size_t(g)].gemvs)) {
      const auto &p = program.projections[size_t(projectionsOf[size_t(g)][j])];
      llvm::json::Array parts;
      for (int part : p.streamPart) parts.push_back(part);
      gemvs.push_back(llvm::json::Object{{"name", gemv.name}, {"n", gemv.n}, {"k", gemv.k},
                                         {"streams", p.streams()}, {"sweeps", p.sweeps()},
                                         {"stream_partitions", std::move(parts)}});
    }
    pimInfo.push_back(llvm::json::Object{{"group", g}, {"rank", rankOf[size_t(g)]},
                                         {"commands", pimPhases[size_t(g)].second}, {"gemvs", std::move(gemvs)}});
  }
  llvm::json::Object metadata{{"schema", "npu.hybrid.package.v1"},
                              {"lp6_bytes", int64_t(options.lp6Bytes)},
                              {"image_bytes", int64_t(imageBytes)},
                              {"dram_scratch_base", int64_t(scratchBase)},
                              {"dram_end", int64_t(dramEnd)},
                              {"kernel_row", program.kernelRow},
                              {"pim_commands", pimCommandCount},
                              {"phases", std::move(phaseInfo)},
                              {"pim_groups", std::move(pimInfo)},
                              {"globals", std::move(globalInfo)},
                              {"outputs", std::move(outputs)}};
  if (!writeJSON(out / "metadata.json", std::move(metadata))) return 1;

  //===--------------------------------------------------------------------===//
  // LP6 이미지 (--inputs)
  //===--------------------------------------------------------------------===//
  if (!options.inputsDir.empty()) {
    Image image;
    if (!image.open(out / "lp6.bin", imageBytes)) { std::cerr << "cannot create lp6.bin\n"; return 1; }
    fs::path inputs(options.inputsDir);
    // 원본 FP16 가중치를 출력 채널별 INT8 로. 전치를 접어 넣은 가중치는 모델 그대로 [N][K].
    auto quantizeSource = [&](int64_t source, uint64_t outputsN, std::vector<int8_t> &q,
                              std::vector<float> &scales) -> bool {
      std::vector<char> raw;
      if (!readFile(inputs / ("arg" + std::to_string(source) + ".bin"), raw) || !outputsN ||
          raw.size() % (2 * outputsN)) {
        std::cerr << "cannot quantize argument " << source << '\n';
        return false;
      }
      const uint64_t inputsK = raw.size() / (2 * outputsN);
      q.resize(outputsN * inputsK);
      scales.resize(outputsN);
      ::npu::quantizeChannels(reinterpret_cast<const uint16_t *>(raw.data()), outputsN, inputsK,
                            pretransposed.count(source) != 0, q.data(), scales.data());
      return true;
    };
    std::map<int64_t, std::vector<float>> scalesOf;
    for (const auto &g : globals) {
      if (g.kind != ::npu::HybridGlobal::Kind::Argument) continue;
      auto type = cast<RankedTensorType>(g.type);
      const uint64_t width = ::npu::elementBytes(type.getElementType());
      const uint64_t cols = uint64_t(type.getShape().back()), rows = uint64_t(type.getNumElements()) / cols;
      std::vector<char> raw;
      bool transposed = pretransposed.count(g.argument) != 0;
      if (auto found = quantized.find(g.argument); found != quantized.end()) {
        auto &scales = scalesOf[found->second.source];
        std::vector<int8_t> q;
        if (found->second.part == "weight") {
          // 선언된 모양 [K, N]. 양자화기는 [N][K] 로 낸다.
          if (!quantizeSource(found->second.source, cols, q, scales)) return 1;
          raw.resize(rows * cols);
          for (uint64_t k = 0; k < rows; ++k)
            for (uint64_t n = 0; n < cols; ++n) raw[k * cols + n] = char(q[n * rows + k]);
        } else {
          if (scales.empty() && !quantizeSource(found->second.source, cols, q, scales)) return 1;
          raw.resize(scales.size() * sizeof(float));
          std::memcpy(raw.data(), scales.data(), raw.size());
        }
        transposed = false;
      } else if (auto held = constants.find(g.argument); held != constants.end()) {
        raw = held->second;
      } else if (!readFile(inputs / ("arg" + std::to_string(g.argument) + ".bin"), raw)) {
        std::cerr << "missing input " << (inputs / ("arg" + std::to_string(g.argument) + ".bin")) << '\n';
        return 1;
      }
      if (raw.size() != g.bytes) {
        std::cerr << "wrong size for " << g.name << ": " << raw.size() << " != " << g.bytes << '\n';
        return 1;
      }
      if (!transposed && !g.packedPanel) {
        if (!image.write(g.address, raw.data(), raw.size())) return 1;
        continue;
      }
      // 전치해 놓거나, 패널 단위로 모아 놓는다(npu-compile --inputs 와 같은 규칙).
      auto source = [&](uint64_t row, uint64_t column) {
        return transposed ? column * rows + row : row * cols + column;
      };
      std::vector<char> laid(raw.size());
      uint64_t to = 0;
      const uint64_t panel = g.packedPanel ? uint64_t(g.packedPanel) : cols;
      if (cols % panel) { std::cerr << "packed panel does not divide " << g.name << '\n'; return 1; }
      for (uint64_t first = 0; first < cols; first += panel)
        for (uint64_t row = 0; row < rows; ++row)
          for (uint64_t column = first; column < first + panel; ++column, ++to)
            std::memcpy(laid.data() + to * width, raw.data() + source(row, column) * width, width);
      if (!image.write(g.address, laid.data(), laid.size())) return 1;
    }
    // PIM 가중치: 32 B 패킷마다 제 뱅크 자리에.
    for (int64_t g = 0; g < groups; ++g)
      for (auto [j, gemv] : llvm::enumerate(plan->pimGroups[size_t(g)].gemvs)) {
        const auto &p = program.projections[size_t(projectionsOf[size_t(g)][j])];
        auto found = quantized.find(gemv.weight);
        if (found == quantized.end() || found->second.part != "weight") {
          std::cerr << gemv.name << " is not a quantized weight\n";
          return 1;
        }
        std::vector<int8_t> q;
        std::vector<float> scales;
        if (!quantizeSource(found->second.source, uint64_t(gemv.n), q, scales)) return 1;
        bool fits = true;
        program.forEachWeightPacket(p, q.data(), [&](uint64_t address, const std::array<uint8_t, 32> &packet) {
          fits &= image.write(address, packet.data(), packet.size());
        });
        if (!fits) { std::cerr << gemv.name << ": PIM packet outside the image\n"; return 1; }
      }
    timed("image");
  }
  std::cout << "wrote hybrid package to " << out << ": " << phases << " NPU phases, " << groups
            << " PIM phases, " << pimCommandCount << " PIM commands, " << words[3] << " records\n";
  return 0;
}
