//===----------------------------------------------------------------------===//
// plena-compile — PLENA 컴파일러 드라이버
//
// linalg 텐서 그래프(.mlir)를 PLENA 칩이 읽는 Unified Program (ISA ver 1.0) 패키지로
// 바꾼다. 변환은 다섯 단계로 나뉘며, 이 파일은 그 다섯을 순서대로 부르는
// 지휘자일 뿐이다. 실제 변환은 전부 lib/ 아래에 있다.
//
//   0막  꺼내기            모델 → linalg IR   python/plena/capture (파이썬)
//   ─────────── 여기서부터 이 파일이 지휘한다 ───────────
//        입력 파싱 + 하드웨어 속성 부착        이 파일 (attachHardware)
//   1막  legalizeGraph     표현 가능성 검사   lib/Transforms/Legalize.cpp
//   2막  tileGraph         타일 루프 생성     lib/Transforms/Tile.cpp
//   3막  placeGraph        버퍼화 + 주소 배정 lib/Transforms/Place.cpp
//   4막  lowerPlacedGraph  명령 생성          lib/Transforms/Lower.cpp
//   5막  EncodePass        ISA 인코딩         lib/Transforms/Encode.cpp
//
// 받는 .mlir 은 0막이 낸 것이거나 test/Integration 의 손으로 쓴 것이다.
// 어느 쪽이든 plena.* 속성이 없는 순수 linalg 이며, 그 속성은 아래
// "명령줄을 IR 속성으로" 구역에서 붙는다.
//
// 명령줄 옵션은 "명령줄을 IR 속성으로" 구역에서 단 한 번 모듈 속성으로
// 옮겨진다. 그 뒤로는 어떤 단계도 argc/argv 를 보지 않으므로, 중간 단계
// 파일만 있으면 --from=<단계> 로 그 지점부터 이어서 컴파일할 수 있다.
//
// 종료 코드: 0 성공, 1 컴파일 실패, 2 사용법 오류.
//===----------------------------------------------------------------------===//
#include "plena/Dialect/PlenaDialect.h"
#include "plena/Transforms/Passes.h"
#include "plena/Transforms/Pipeline.h"
#include "plena/Target/HardwareConfig.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/JSON.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

namespace {

/// 모듈을 .mlir 텍스트로 저장한다. 여기서 나온 텍스트는 그대로 다시
/// 파싱될 수 있어야 하며, --reparse-stages 가 매 단계 그것을 검사한다.
bool save(mlir::ModuleOp module, const std::filesystem::path &path) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path.string(), ec);
  if (ec) {
    std::cerr << ec.message() << '\n';
    return false;
  }
  module.print(out);
  out << '\n';
  out.flush();
  return !out.has_error();
}

/// 명령줄에서 읽은 설정 한 벌.
///
/// 앞의 다섯은 경로와 시작 지점, 뒤의 넷은 켜고 끄는 스위치다. 선언과
/// 사용처가 멀리 떨어져 있어서(saveStages 는 200줄 아래 checkpoint 안에서
/// 쓰인다) 한 덩어리로 묶어 둔다. 사용처에서 options. 이 붙으므로 그 값이
/// 명령줄에서 왔다는 것도 그 자리에서 드러난다.
struct Options {
  /// 입력 .mlir 경로. '-' 로 시작하지 않는 첫 자유 인자가 여기 들어온다.
  std::string input;

  /// -o : 패키지와 단계 파일이 들어갈 디렉터리.
  std::string output;

  /// --settings : 하드웨어 설정 .toml 경로. 시뮬레이터가 읽는 그 파일이다.
  std::string settings;

  /// --from : 어느 단계부터 시작할지. graph|legal|tiled|placed|target 중 하나.
  std::string fromStage;

  /// --inputs : arg0.bin, arg1.bin ... 이 들어 있는 디렉터리. 주면 실행용
  /// DRAM 이미지(hbm.bin)까지 조립한다. 컴파일 자체에는 필요 없다.
  std::string inputsDir;

  /// --fp16 : 중간 계산에서 FP16 반올림을 허용한다. 수치 정책이라 사용자가
  /// 명시해야 하며, graph 입력에는 필수다 — 없으면 1막이 거부한다.
  bool fp16 = false;
  bool timeStages = false;

  /// --reciprocal-division : a/b 를 a * (1/b) 로 낮춘다. 하드웨어에 나눗셈
  /// 명령이 없어 이 방법뿐이지만, 명령을 둘 쓰고 정확도를 조금 잃으므로
  /// 켜고 끄는 선택으로 둔다.
  bool reciprocalDivision = false;

  /// --readback : 끝에 출력을 L2로 되읽는 GDMA를 붙인다. NPU_Simulator main은
  /// DRAM을 덤프하지 않으므로, 검증은 l2_sram_dump.bin에서 결과를 읽는다.
  /// 주소는 metadata.json의 outputs[].readback_l2_address에 적힌다.
  bool readback = false;

  /// --save-stages : 단계마다 .mlir 을 출력 디렉터리에 남긴다.
  bool saveStages = false;

  /// --reparse-stages : 남긴 파일을 도로 읽어 그 위에서 이어 간다. 속성이
  /// 텍스트 왕복에서 사라지지 않는지 매 단계 확인하게 된다. saveStages 를
  /// 함께 켠다 — 다시 읽으려면 파일이 먼저 있어야 하기 때문이다.
  bool reparseStages = false;
};

} // namespace

int main(int argc, char **argv) {
  //===--------------------------------------------------------------------===//
  // 명령줄 읽기
  //===--------------------------------------------------------------------===//
  Options options;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help") {
      std::cout << "usage: plena-compile --from=graph|legal|tiled|placed|target input.mlir --settings hardware.toml -o directory\n"
                   "  --fp16              allow intermediate FP16 rounding (required for graph input)\n"
                   "  --reciprocal-division  lower a/b as a * (1/b); the hardware has no divide\n"
                   "  --inputs directory  pack contiguous FP16 arg0.bin, arg1.bin, ... into hbm.bin\n"
                   "  --readback          copy outputs back into L2 at the end (read from l2_sram_dump.bin)\n"
                   "  --time-stages        report pass wall times\n"
                   "  --save-stages | --reparse-stages\n";
      return 0;
    }

    // rfind(s, 0) == 0 은 "s 로 시작하는가"를 묻는 관용구다.
    if (arg.rfind("--from=", 0) == 0) {
      options.fromStage = arg.substr(7);
    } else if (arg == "--time-stages") {
      options.timeStages = true;
    } else if (arg == "--fp16") {
      options.fp16 = true;
    } else if (arg == "--reciprocal-division") {
      options.reciprocalDivision = true;
    } else if (arg == "--readback") {
      options.readback = true;
    } else if (arg == "--save-stages") {
      options.saveStages = true;
    } else if (arg == "--reparse-stages") {
      // 다시 읽으려면 파일이 먼저 있어야 하므로 저장도 함께 켠다.
      options.reparseStages = true;
      options.saveStages = true;
    } else if (arg == "-o" && i + 1 < argc) {
      options.output = argv[++i];
    } else if (arg == "--settings" && i + 1 < argc) {
      options.settings = argv[++i];
    } else if (arg == "--inputs" && i + 1 < argc) {
      options.inputsDir = argv[++i];
    } else if (options.input.empty() && !arg.empty() && arg[0] != '-') {
      // '-' 로 시작하지 않는 첫 인자만 입력 파일이 된다. 두 번째 자유 인자는
      // 이 조건에서 떨어져 아래 거부로 간다 — 오타를 조용히 삼키지 않는다.
      options.input = arg;
    } else {
      std::cerr << "invalid argument: " << arg << '\n';
      return 2;
    }
  }

  //===--------------------------------------------------------------------===//
  // 시작 단계 결정
  //
  // 단계 이름이 여기 한 번만 나열되고, 아래 계단식 진입 판정이 이 순서에서
  // 나온다. --from 값이 목록에 없으면 start 가 end() 라 곧바로 걸린다.
  //===--------------------------------------------------------------------===//
  const std::vector<std::string> stageNames{"graph", "legal", "tiled", "placed", "target"};
  auto start = std::find(stageNames.begin(), stageNames.end(), options.fromStage);

  if (start == stageNames.end() || options.input.empty() || options.output.empty() || options.settings.empty()) {
    std::cerr << "requires --from=target or --from=graph|legal|tiled|placed, input.mlir --settings hardware.toml -o directory\n";
    return 2;
  }

  //===--------------------------------------------------------------------===//
  // MLIR 준비와 입력 파싱
  //===--------------------------------------------------------------------===//
  mlir::DialectRegistry registry;

  // 다이얼렉트와 외부 인터페이스 모델 등록. 하나라도 빠지면
  // MLIR 은 진단이 아니라 LLVM ERROR 로 abort 한다 — Graph.cpp 를 볼 것.
  ::plena::registerPipelineDialects(registry);

  mlir::MLIRContext context(registry);

  // 타일링과 버퍼화가 라이브러리 코드 안에서 scf/tensor/memref op 을 만든다.
  // 그 시점을 제어할 수 없으므로 지연 로딩에 기대지 않고 미리 전부 올린다.
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(options.input, &context);
  if (!module || mlir::failed(mlir::verify(*module)))
    return 1;

  //===--------------------------------------------------------------------===//
  // 입력 파일이 정말 --from 이 말하는 단계인지 확인
  //
  // 단계 파일에는 그것을 만든 단계가 도장처럼 찍혀 있다(plena.stage).
  // --from=placed 라면서 실제로는 02-tiled.mlir 을 주면, 아직 하지 않은
  // 작업을 건너뛴 채 진행되어 결과물이 조용히 망가진다. 여기서 막는다.
  //
  // graph 만 예외인 이유: 그 파일은 이 컴파일러를 아직 한 번도 거치지
  // 않았으므로 도장이 없다. 0막 캡처(python/plena/capture)가 torch-mlir 로
  // 뽑은 linalg IR 이거나 test/Integration 의 손으로 쓴 .mlir 인데, 둘 다
  // plena.* 속성을 모른다. 도장은 아래 "명령줄을 IR 속성으로" 구역에서
  // 이 파일이 직접 찍어 준다.
  //
  // plena.schema 는 단계 파일 형식의 버전이다. 지금은 항상 1이고, 형식이
  // 바뀌면 올려서 옛 파일을 거부하게 된다.
  //===--------------------------------------------------------------------===//
  auto stage = (*module)->getAttrOfType<mlir::StringAttr>("plena.stage");
  auto schema = (*module)->getAttrOfType<mlir::IntegerAttr>("plena.schema");

  if (options.fromStage != "graph" &&
      (!stage || stage.getValue() != options.fromStage || !schema || schema.getInt() != 1)) {
    std::cerr << "input stage/schema does not match --from\n";
    return 1;
  }

  //===--------------------------------------------------------------------===//
  // 하드웨어 설정 읽기
  //===--------------------------------------------------------------------===//
  // Result<T> 는 Target/ 의 자체 타입이다. Target/ 에는 MLIR 의존이 없어
  // LogicalResult 를 쓸 수 없다.
  auto hardware = ::plena::loadHardwareConfig(options.settings);
  if (!hardware) {
    std::cerr << hardware.error << '\n';
    return 1;
  }
  const auto &hw = hardware.value;

  //===--------------------------------------------------------------------===//
  // 명령줄을 IR 속성으로
  //
  // 이 구역이 이 파일의 핵심이다. 여기서 옵션이 모듈 속성이 되고, 이후
  // 다섯 단계는 argc/argv 를 전혀 모른다. 중간 단계 파일이 자기 정책을
  // 들고 다니므로 --from 재진입이 성립한다.
  //===--------------------------------------------------------------------===//
  if (options.fromStage == "graph") {
    // cores / l1_bytes / l2_bytes / dram_bytes / hardware / schema 를 붙인다.
    ::plena::attachHardware(*module, hw);

    (*module)->setAttr("plena.stage", mlir::StringAttr::get(&context, "graph"));

    // FP16 중간 반올림은 수치 정책이라 사용자가 명시해야 한다.
    // 빠뜨리면 legalizeGraph 가 거부한다.
    if (options.fp16)
      (*module)->setAttr("plena.numerical", mlir::StringAttr::get(&context, "fp16"));

    if (options.reciprocalDivision)
      (*module)->setAttr("plena.division", mlir::StringAttr::get(&context, "reciprocal"));

    if (options.readback)
      (*module)->setAttr("plena.readback", mlir::UnitAttr::get(&context));
  }

  //===--------------------------------------------------------------------===//
  // 용량 검사
  //
  // --from=graph 면 방금 붙인 값이라 항상 통과한다. 의미가 있는 것은 중간
  // 단계 재진입일 때다 — 큰 하드웨어로 만든 단계 파일을 작은 설정으로
  // 이어서 컴파일하는 것을 막는다.
  //
  // 3막을 지난 파일에서는 이 속성들이 용량이 아니라 실제 사용량(high water)
  // 으로 덮여 있다. 사용량은 언제나 용량 이하이므로 검사는 그대로 성립한다.
  //===--------------------------------------------------------------------===//
  struct CapacityLimit {
    const char *attribute;
    uint64_t hardwareMaximum;
  };
  const CapacityLimit limits[] = {
      {"plena.cores", hw.cores},
      {"plena.l1_bytes", hw.l1Bytes},
      {"plena.l2_bytes", hw.l2Bytes},
      {"plena.dram_bytes", hw.dramBytes},
  };

  for (const auto &limit : limits) {
    auto declared = (*module)->getAttrOfType<mlir::IntegerAttr>(limit.attribute);
    if (!declared || declared.getInt() <= 0 ||
        uint64_t(declared.getInt()) > limit.hardwareMaximum) {
      std::cerr << limit.attribute << " exceeds hardware capacity or is missing\n";
      return 1;
    }
  }

  //===--------------------------------------------------------------------===//
  // 설정 지문 대조
  //
  // 지문은 설정 파일 전문의 FNV-1a 다(파싱된 값이 아니라 원문). 캐시 무효화
  // 식별자이며 보안 해시가 아니다. 다른 설정으로 만든 단계 파일을 거부한다.
  //===--------------------------------------------------------------------===//
  auto fingerprint = (*module)->getAttrOfType<mlir::StringAttr>("plena.hardware");
  if (fingerprint && fingerprint.getValue() != hw.fingerprint) {
    std::cerr << "stage hardware fingerprint differs from settings\n";
    return 1;
  }
  (*module)->setAttr("plena.hardware", mlir::StringAttr::get(&context, hw.fingerprint));

  //===--------------------------------------------------------------------===//
  // 출력 디렉터리
  //===--------------------------------------------------------------------===//
  std::filesystem::path outputDir(options.output);
  std::error_code ec;
  std::filesystem::create_directories(outputDir, ec);
  if (ec) {
    std::cerr << ec.message() << '\n';
    return 1;
  }

  /// 한 단계가 끝날 때마다: 검증하고, 요청하면 저장하고, 요청하면 다시 읽는다.
  ///
  /// 재파싱은 module 을 통째로 교체한다([&] 캡처라 바깥 변수가 바뀐다).
  /// 즉 --reparse-stages 를 켜면 이후 단계는 디스크에서 다시 읽은 모듈 위에서
  /// 돌고, 속성 하나라도 텍스트 왕복에서 사라지면 여기서 드러난다.
  auto checkpoint = [&](const char *name) -> bool {
    if (mlir::failed(mlir::verify(*module)))
      return false;

    if (options.saveStages && !save(*module, outputDir / name))
      return false;

    if (options.reparseStages) {
      module = mlir::parseSourceFile<mlir::ModuleOp>((outputDir / name).string(), &context);
      if (!module || mlir::failed(mlir::verify(*module)))
        return false;
    }
    return true;
  };

  //===--------------------------------------------------------------------===//
  // 파이프라인
  //
  // firstStage 가 시작 단계의 인덱스다: graph=0, legal=1, tiled=2, placed=3,
  // target=4. 각 구역의 조건이 누적이라 --from 이 가리키는 지점부터 들어간다.
  //===--------------------------------------------------------------------===//
  auto timed = [&](const char *name, auto action) {
    auto begin = std::chrono::steady_clock::now();
    auto result = action();
    if (options.timeStages)
      std::cerr << "timing " << name << " "
                << std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()
                << " s\n";
    return result;
  };
  size_t firstStage = std::distance(stageNames.begin(), start);

  // 단계마다 PassManager 하나로 그 단계의 패스를 돌린다(lib/Transforms/Passes.cpp).
  // 한 번에 다 넣지 않는 이유: 단계 사이에서 checkpoint 가 저장·재파싱을 하고,
  // 재파싱하면 module 이 다른 객체로 바뀌기 때문이다. PassManager 는 패스가 끝날
  // 때마다 IR 검증을 한다.
  auto runPass = [&](std::unique_ptr<mlir::Pass> pass) {
    mlir::PassManager pm(&context);
    pm.addPass(std::move(pass));
    return pm.run(*module);
  };

  // 1막 — 받을 수 있는 그래프인지 검사한다. 실제 일의 대부분은 그 앞의
  // normalizeGraph 가 하는 재작성들이다.
  if (firstStage == 0) {
    if (!checkpoint("00-imported.mlir"))
      return 1;
    if (mlir::failed(timed("legalize", [&] { return runPass(::plena::createLegalizePass()); })))
      return 1;
    if (!checkpoint("01-legal.mlir"))
      return 1;
  }

  // 2막 — 타일 계획을 고르고 타일 루프를 세운다. 아직 전부 텐서 레벨이다.
  // 하드웨어가 필요한 이유: 타일 크기가 L1/L2 용량에서 나온다.
  if (firstStage <= 1) {
    if (mlir::failed(timed("tile", [&] { return runPass(::plena::createTilePass(hw)); })))
      return 1;
    if (!checkpoint("02-tiled.mlir"))
      return 1;
  }

  // 3막 — 버퍼화(tensor → memref), 생존 구간 분석(mlir::Liveness +
  // BufferViewFlowAnalysis), 주소 배정. 여기서 텐서가 사라진다.
  if (firstStage <= 2) {
    if (mlir::failed(timed("place", [&] { return runPass(::plena::createPlacePass(hw)); })))
      return 1;
    if (!checkpoint("03-placed.mlir"))
      return 1;
  }

  // 4막 — 명령 생성. 모듈 본문을 비우고 plena.dma / plena.core_block 으로
  // 다시 쓴다(치환이 아니다).
  if (firstStage <= 3) {
    if (mlir::failed(timed("lower", [&] { return runPass(::plena::createLowerPass()); })))
      return 1;
  }

  // 여기만 checkpoint 를 쓰지 않는다. 파일 이름이 03-target.mlir 인데, 이는
  // --from=target 의 진입점이며 3막 결과인 03-placed.mlir 과는 별개다.
  // 손으로 쓴 타깃 IR 로 S1 만 돌리는 경로가 이 이름을 쓴다.
  if (options.saveStages && !save(*module, outputDir / "03-target.mlir"))
    return 1;
  if (options.reparseStages) {
    module = mlir::parseSourceFile<mlir::ModuleOp>((outputDir / "03-target.mlir").string(), &context);
    if (!module || mlir::failed(mlir::verify(*module)))
      return 1;
  }

  // 5막 — 타깃 op 들을 32비트 워드로 인코딩한다.
  if (mlir::failed(timed("encode", [&] { return runPass(::plena::createEncodePass()); })))
    return 1;

  if (options.saveStages && !save(*module, outputDir / "04-isa.mlir"))
    return 1;
  if (options.reparseStages) {
    module = mlir::parseSourceFile<mlir::ModuleOp>((outputDir / "04-isa.mlir").string(), &context);
    if (!module || mlir::failed(mlir::verify(*module)))
      return 1;
  }

  //===--------------------------------------------------------------------===//
  // 패키지 쓰기
  //
  // EncodePass 가 모듈 본문을 비우고 plena.program 하나만 남겼으므로,
  // 정확히 하나여야 한다.
  //===--------------------------------------------------------------------===//
  auto programs = module->getOps<mlir::plena::ProgramOp>();
  if (std::distance(programs.begin(), programs.end()) != 1)
    return 1;
  auto program = *programs.begin();

  // 워드를 리틀엔디안으로 직접 쓴다. 호스트 엔디안에 기대지 않으므로 어느
  // 기계에서 컴파일해도 같은 program.bin 이 나온다. 워드를 int32_t 로 들고
  // 다니는 것은 DenseI32ArrayAttr 가 부호 있는 정수만 받기 때문이다.
  std::ofstream binary(outputDir / "program.bin", std::ios::binary);
  for (int32_t word : program.getWords()) {
    uint32_t bits = static_cast<uint32_t>(word);
    for (unsigned byte = 0; byte < 4; ++byte)
      binary.put(char(bits >> (8 * byte)));
  }
  binary.close();

  // system.json 은 시뮬레이터가 읽는 Unified Program (ISA ver 1.0) 매니페스트,
  // metadata.json 은 이 컴파일러가 만든 ABI 설명이다. 소비자가 다르다.
  std::ofstream manifest(outputDir / "system.json");
  manifest << program.getManifest().str();
  manifest.close();

  if (!binary || !manifest) {
    std::cerr << "failed to write package\n";
    return 1;
  }

  //===--------------------------------------------------------------------===//
  // 실행용 DRAM 이미지 조립 (--inputs)
  //
  // 여기부터는 컴파일이 아니라 실행 준비다. 손으로 쓴 타깃 IR 경로에는
  // 패키지 메타데이터가 없으므로 --inputs 를 받을 수 없다.
  //===--------------------------------------------------------------------===//
  auto metadata = (*module)->getAttrOfType<mlir::StringAttr>("plena.metadata");

  if (!metadata && !options.inputsDir.empty()) {
    std::cerr << "--inputs requires graph package metadata\n";
    return 1;
  }

  if (metadata) {
    std::ofstream meta(outputDir / "metadata.json");
    meta << metadata.getValue().str();
    meta.close();
    if (!meta)
      return 1;
  }

  if (metadata && !options.inputsDir.empty()) {
    auto parsed = llvm::json::parse(metadata.getValue());
    if (!parsed || !parsed->getAsObject()) {
      std::cerr << "invalid package metadata\n";
      return 1;
    }
    auto *packageInfo = parsed->getAsObject();

    auto dramBytes = packageInfo->getInteger("dram_bytes");
    auto *args = packageInfo->getArray("arguments");
    if (!dramBytes || *dramBytes <= 0 || !args)
      return 1;

    // 컴파일러가 전치를 접어 없앤 가중치는 모델이 가진 그대로 저장되어 있다.
    // 프로그램이 읽는 것은 이미지이므로, 뒤집는 일은 여기서 한다.
    std::set<int64_t> flipped;
    if (auto *list = packageInfo->getArray("pretransposed"))
      for (const auto &value : *list)
        if (auto index = value.getAsInteger())
          flipped.insert(*index);

    // 1막이 상수 텐서를 인자로 승격하면서 그 바이트를 패키지에 넣었다.
    // 그러므로 상수는 패키지에서 나온다 — 아무도 요구하지 않은 파일이 아니라.
    std::map<int64_t, std::vector<char>> held;
    if (auto *list = packageInfo->getArray("constants")) {
      for (const auto &value : *list) {
        auto *constant = value.getAsObject();
        if (!constant)
          return 1;

        auto index = constant->getInteger("index");
        auto encoded = constant->getString("base64");
        if (!index || !encoded)
          return 1;

        std::vector<char> decoded;
        if (llvm::decodeBase64(*encoded, decoded)) {
          std::cerr << "invalid constant payload\n";
          return 1;
        }
        held[*index] = std::move(decoded);
      }
    }

    // 마지막 바이트 위치로 건너뛰어 0 하나를 쓰면 파일 크기가 dram_bytes 가
    // 되고 그 사이는 파일시스템이 0 으로 채운다. DRAM 이미지 전체를 메모리에
    // 올리지 않고 만드는 방법이다.
    std::ofstream image(outputDir / "hbm.bin", std::ios::binary);
    image.seekp(*dramBytes - 1);
    image.put(0);

    for (const auto &value : *args) {
      auto *argument = value.getAsObject();
      if (!argument)
        return 1;

      auto index = argument->getInteger("index");
      auto address = argument->getInteger("address");
      auto rows = argument->getInteger("rows");
      auto cols = argument->getInteger("cols");
      if (!index || !address || !rows || !cols)
        return 1;

      auto dtype = argument->getString("dtype");
      const uint64_t width = dtype && *dtype == "float32" ? 4 : 2;
      const uint64_t expected = uint64_t(*rows) * uint64_t(*cols) * width;
      auto path = std::filesystem::path(options.inputsDir) / ("arg" + std::to_string(*index) + ".bin");
      auto panel = argument->getInteger("packed");
      std::vector<char> raw;
      // Constants and external inputs obey the same physical-layout contract.
      if (auto found = held.find(*index); found != held.end()) {
        raw = found->second;
      } else {
        auto size = std::filesystem::file_size(path, ec);
        if (ec || size != expected) {
          std::cerr << "wrong input size: " << path << '\n';
          return 1;
        }
        std::ifstream data(path, std::ios::binary);
        if (!flipped.count(*index) && !panel) {
          image.seekp(*address);
          image << data.rdbuf();
          if (!data || !image) return 1;
          continue;
        }
        raw.resize(size);
        data.read(raw.data(), std::streamsize(size));
        if (!data) return 1;
      }
      if (raw.size() != expected) {
        std::cerr << "wrong constant size for argument " << *index << '\n';
        return 1;
      }
      image.seekp(*address);

      // 패널 단위로 모아 달라고 한 가중치. 그래야 패널 하나가 전송 하나가 된다.

      if (!flipped.count(*index) && !panel) {
        image.write(raw.data(), std::streamsize(raw.size()));
      } else {
        const int64_t rowCount = *rows;
        const int64_t columnCount = *cols;

        // 파일에는 모델이 들고 있는 대로 들어 있다. 전치를 접어 넣은 가중치는
        // rows 개짜리 행이 cols 줄이므로 반대로 읽는다.
        auto source = [&](int64_t row, int64_t column) {
          return flipped.count(*index) ? size_t(column * rowCount + row)
                                       : size_t(row * columnCount + column);
        };

        std::vector<char> laid(raw.size());
        size_t to = 0;

        if (!panel) {
          for (int64_t row = 0; row < rowCount; ++row)
            for (int64_t column = 0; column < columnCount; ++column, ++to) {
              // 원소를 값으로 해석하지 않고 바이트 그대로 옮긴다.
              size_t from = source(row, column) * width;
              std::copy_n(raw.data() + from, width, laid.data() + to * width);
            }
        } else {
          if (*panel <= 0 || columnCount % *panel) {
            std::cerr << "packed panel does not divide the weight: " << path << '\n';
            return 1;
          }
          for (int64_t first = 0; first < columnCount; first += *panel)
            for (int64_t row = 0; row < rowCount; ++row)
              for (int64_t column = first; column < first + *panel; ++column, ++to) {
                size_t from = source(row, column) * width;
                std::copy_n(raw.data() + from, width, laid.data() + to * width);
              }
        }
        image.write(laid.data(), std::streamsize(laid.size()));
      }

      if (!image)
        return 1;
    }

    image.close();
    if (!image)
      return 1;
  }

  std::cout << "wrote Unified Program (ISA ver 1.0) package to " << outputDir << '\n';
  return 0;
}
