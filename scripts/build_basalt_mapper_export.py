#!/usr/bin/env python3
"""Build the headless map exporter with the configured Basalt Ninja toolchain."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basalt-source", type=Path, default=root / "third_party/basalt_monado")
    args = parser.parse_args()
    source = args.basalt_source.resolve()
    build = source / "build"
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(entry for entry in entries if Path(entry["file"]) == source / "src/mapper.cpp")
    command = shlex.split(entry["command"])
    obj = build / "basalt_mapper_export.cpp.o"
    command[command.index("-o") + 1] = str(obj)
    command[command.index("-c") + 1] = str(root / "tools/basalt_mapper_export.cpp")
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    env["CCACHE_DIR"] = str(build / ".ccache")
    for name in ("LD_LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH"):
        env.pop(name, None)
    print("Compiling headless Basalt map exporter", flush=True)
    subprocess.run(command, cwd=entry["directory"], env=env, check=True)
    commands = subprocess.check_output(["/usr/bin/ninja", "-t", "commands", "basalt_mapper"], cwd=build, env=env, text=True)
    segments = commands.splitlines()[-1].split(" && ")
    link = shlex.split(next(segment for segment in segments if " -o basalt_mapper " in segment))
    link[link.index("-o") + 1] = str(build / "basalt_mapper_export")
    link = [str(obj) if item == "CMakeFiles/basalt_mapper.dir/src/mapper.cpp.o" else
            item.replace("CMakeFiles/basalt_mapper.dir/thirdparty/vit/", "CMakeFiles/basalt_vio.dir/thirdparty/vit/")
            for item in link]
    subprocess.run(link, cwd=build, env=env, check=True)
    print(build / "basalt_mapper_export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
