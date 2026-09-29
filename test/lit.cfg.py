import os
import shlex

import lit.formats

config.name = "NPU"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.excludes = ["Integration"]
config.test_source_root = os.path.dirname(__file__)
config.substitutions.extend([
    ("%npu-opt", shlex.quote(os.path.join(config.npu_tools_dir, "npu-opt"))),
    ("%FileCheck", shlex.quote(config.filecheck)),
    ("%npu-asm", shlex.quote(os.path.join(config.npu_tools_dir, "npu-asm"))),
    ("%npu-compile", shlex.quote(os.path.join(config.npu_tools_dir, "npu-compile"))),
])
