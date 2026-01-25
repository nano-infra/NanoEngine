import os
import re

TARGET_DIR = "/home/majinming/NanoDeployPython/NanoDeploy/csrc"


def refactor_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    original_content = content

    # 1. Replace include logging
    content = content.replace(
        '#include "nanodeploy/logging.h"', '#include "nanocommon/logging.h"'
    )

    # 2. Replace macros NANODEPLOY_ -> NANOINFRA_
    # Use regex to be safe or just string replace if robust
    # NANODEPLOY_LOG_INFO -> NANOINFRA_LOG_INFO
    # NANODEPLOY_ASSERT -> NANOINFRA_ASSERT
    content = re.sub(r"NANODEPLOY_", "NANOINFRA_", content)

    # 3. Replace namespace nanodeploy -> namespace nanoinfra
    content = re.sub(r"namespace nanodeploy", "namespace nanoinfra", content)

    # 4. Replace nanodeploy:: -> nanoinfra::
    # But avoid replacing includes like "nanodeploy/filename.h"
    # Negative lookbehind/ahead or specific contexts?
    # Simply replacing `nanodeploy::` is usually safe.
    content = content.replace("nanodeploy::", "nanoinfra::")

    # Check for "nanodeploy" string usage that implies package name?
    # e.g. PYBIND11_MODULE(nanodeploy, m) -> PYBIND11_MODULE(nanoinfra, m)
    # Be careful not to break "nanodeploy" directory paths in strings if used.

    # Let's handle PYBIND11_MODULE specific case
    content = re.sub(
        r"PYBIND11_MODULE\(nanodeploy", "PYBIND11_MODULE(nanoinfra", content
    )

    if content != original_content:
        print(f"Refactoring {filepath}")
        with open(filepath, "w") as f:
            f.write(content)


for root, dirs, files in os.walk(TARGET_DIR):
    for file in files:
        if file.endswith((".h", ".cpp", ".hpp", ".c", ".cc", ".cu", ".cuh")):
            refactor_file(os.path.join(root, file))
