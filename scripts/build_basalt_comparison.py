#!/usr/bin/env python3
"""Build original Basalt and matching headless map exporters without editing either checkout."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args()
    output = args.output.resolve()
    env = os.environ.copy()
    for key in ("CONDA_PREFIX", "CMAKE_PREFIX_PATH", "LD_LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH"):
        env.pop(key, None)
    env["PATH"] = "/usr/bin:/bin"
    env["CCACHE_DIR"] = str(output / ".ccache")
    headers = output / "deps/basalt-headers"
    revision = subprocess.check_output(["git", "-C", str(headers), "rev-parse", "HEAD"], text=True).strip()
    if revision != "aa441ba3e51050c47ba1902537792a2e4db7e43d":
        raise ValueError("original basalt-headers must match its vcpkg port's pinned revision")
    build = output / "build_original"
    overlay = output / "deps/tbb_alignment"
    for name in ("tbb", "oneapi/tbb"):
        if not (overlay / name).exists():
            shutil.copytree(Path("/usr/include") / name, overlay / name)
    header = overlay / "oneapi/tbb/detail/_concurrent_unordered_base.h"
    text = header.read_text()
    old = "std::aligned_storage<sizeof(value_type)>::type"
    new = "std::aligned_storage<sizeof(value_type), alignof(value_type)>::type"
    if old in text:
        header.write_text(text.replace(old, new))
    elif new not in text:
        raise ValueError("unexpected oneTBB header; inspect alignment before building")
    subprocess.run(["cmake", "-S", str(ROOT / "tools/basalt_local_comparison"), "-B", str(build), "-G", "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release", f"-DBASALT_HEADERS={headers}",
                    f"-DTBB_ALIGNMENT_OVERLAY={overlay}",
                    "-Dfmt_DIR=/usr/lib/x86_64-linux-gnu/cmake/fmt"], env=env, check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "basalt_vio", "basalt_mapper_export", "-j", str(args.jobs)], env=env, check=True)
    source = ROOT / "third_party/basalt_monado"
    build = source / "build"
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(e for e in entries if Path(e["file"]) == source / "src/mapper.cpp")
    command = shlex.split(entry["command"])
    obj = output / "monado_mapper_comparison.o"
    command[command.index("-o") + 1] = str(obj)
    command[command.index("-c") + 1] = str(ROOT / "tools/basalt_local_comparison/mapper_export.cpp")
    subprocess.run(command, cwd=entry["directory"], env=env, check=True)
    commands = subprocess.check_output(["/usr/bin/ninja", "-t", "commands", "basalt_mapper"], cwd=build, env=env, text=True)
    segment = next(s for s in commands.splitlines()[-1].split(" && ") if " -o basalt_mapper " in s)
    link = shlex.split(segment)
    link[link.index("-o") + 1] = str(output / "basalt_monado_mapper_export")
    link = [str(obj) if item == "CMakeFiles/basalt_mapper.dir/src/mapper.cpp.o" else
            item.replace("CMakeFiles/basalt_mapper.dir/thirdparty/vit/", "CMakeFiles/basalt_vio.dir/thirdparty/vit/") for item in link]
    subprocess.run(link, cwd=build, env=env, check=True)
    print("Both comparison map exporters built", flush=True)


if __name__ == "__main__":
    main()
