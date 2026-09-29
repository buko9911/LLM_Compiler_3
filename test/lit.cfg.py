import os
import shlex

import lit.formats

config.name = "PLENA"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.excludes = ["Integration"]
config.test_source_root = os.path.dirname(__file__)
config.substitutions.extend([
    ("%plena-opt", shlex.quote(os.path.join(config.plena_tools_dir, "plena-opt"))),
    ("%FileCheck", shlex.quote(config.filecheck)),
    ("%plena-asm", shlex.quote(os.path.join(config.plena_tools_dir, "plena-asm"))),
    ("%plena-compile", shlex.quote(os.path.join(config.plena_tools_dir, "plena-compile"))),
])
