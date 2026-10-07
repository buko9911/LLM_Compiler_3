// One compiler executable; input contracts select a pipeline, all share NPUTarget.
#include <iostream>
#include <string>
#include <vector>

int compileGraph(int argc, char **argv);
int compileHybridGraph(int argc, char **argv);
int compileDecode(int argc, char **argv);

int main(int argc, char **argv) {
  std::string pipeline;
  bool manifest = false, help = false;
  std::vector<char *> args{argv[0]};
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--pipeline" || arg.rfind("--pipeline=", 0) == 0) {
      if (!pipeline.empty()) {
        std::cerr << "specify --pipeline only once\n";
        return 2;
      }
      if (arg == "--pipeline") {
        if (++i == argc) { std::cerr << "--pipeline needs a value\n"; return 2; }
        pipeline = argv[i];
      } else pipeline = arg.substr(11);
      if (pipeline != "graph" && pipeline != "graph-hybrid" && pipeline != "decode") {
        std::cerr << "--pipeline must be graph, graph-hybrid or decode\n";
        return 2;
      }
    } else {
      manifest |= arg == "--manifest" || arg.rfind("--manifest=", 0) == 0;
      help |= arg == "--help";
      args.push_back(argv[i]);
    }
  }
  if (pipeline.empty() && help && !manifest) {
    std::cout << "LLM_Compiler_3: npu-compile\n"
                 "  --pipeline=decode        Qwen3/Llama explicit-state graph + --manifest + --spec\n"
                 "  --pipeline=graph         general tensor/linalg graph + --from + --settings\n"
                 "  --pipeline=graph-hybrid  captured W8A8 graph + --settings + --config\n"
                 "With --manifest, decode is selected automatically; otherwise graph.\n"
                 "Use --pipeline=<name> --help for that input contract's options.\n";
    return 0;
  }
  if (pipeline.empty()) pipeline = manifest ? "decode" : "graph";
  if (manifest && pipeline != "decode") {
    std::cerr << "--manifest requires --pipeline=decode\n";
    return 2;
  }
  const int count = int(args.size());
  args.push_back(nullptr);
  if (pipeline == "decode") return compileDecode(count, args.data());
  if (pipeline == "graph-hybrid") return compileHybridGraph(count, args.data());
  return compileGraph(count, args.data());
}
