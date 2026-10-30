// Decode compiler entry point. Keep --pipeline=decode for existing scripts.
#include <iostream>
#include <string>
#include <vector>

int compileDecode(int argc, char **argv);

int main(int argc, char **argv) {
  bool pipelineSeen = false;
  std::vector<char *> args{argv[0]};
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--pipeline" || arg.rfind("--pipeline=", 0) == 0) {
      if (pipelineSeen) {
        std::cerr << "specify --pipeline only once\n";
        return 2;
      }
      pipelineSeen = true;
      std::string pipeline;
      if (arg == "--pipeline") {
        if (++i == argc) {
          std::cerr << "--pipeline needs a value\n";
          return 2;
        }
        pipeline = argv[i];
      } else {
        pipeline = arg.substr(11);
      }
      if (pipeline != "decode") {
        std::cerr << "--pipeline must be decode\n";
        return 2;
      }
    } else {
      args.push_back(argv[i]);
    }
  }
  const int count = int(args.size());
  args.push_back(nullptr);
  return compileDecode(count, args.data());
}
