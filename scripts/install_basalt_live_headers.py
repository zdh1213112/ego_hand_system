#!/usr/bin/env python3
"""Install pinned header-only dependencies for the native Basalt live adapter."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

DEPENDENCIES = (
    ('basalt-headers', 'aa441ba3e51050c47ba1902537792a2e4db7e43d',
     'https://gitlab.com/VladyslavUsenko/basalt-headers/-/archive/aa441ba3e51050c47ba1902537792a2e4db7e43d/basalt-headers-aa441ba3e51050c47ba1902537792a2e4db7e43d.tar.gz',
     '0024bf2601bd8c16ff6e073ae4f33e4907a081fe4e5e9cbd59344686afaaa766',
     'include/basalt/calibration/calibration.hpp'),
    ('Sophus', '1.22.10', 'https://codeload.github.com/strasdat/Sophus/tar.gz/refs/tags/1.22.10',
     'eb1da440e6250c5efc7637a0611a5b8888875ce6ac22bf7ff6b6769bbc958082', 'sophus/se3.hpp'),
    ('cereal', 'v1.3.2', 'https://codeload.github.com/USCiLab/cereal/tar.gz/refs/tags/v1.3.2',
     '16a7ad9b31ba5880dac55d62b5d6f243c3ebc8d46a3514149e56b5e7ea81f85f', 'include/cereal/cereal.hpp'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'third_party/basalt_live_deps')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output/'manifest.json'
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    by_name = {r['name']:r for r in records}
    for name, version, url, expected, header in DEPENDENCIES:
        destination = args.output/name
        if destination.exists():
            if by_name.get(name, {}).get('sha256') == expected and (destination/header).is_file():
                print('Verified installed dependency:', name, version, flush=True)
                continue
            raise FileExistsError(f'Unverified dependency directory: {destination}')
        print('Downloading:', name, version, flush=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f'{name} SHA256 mismatch: {actual}')
        with tempfile.TemporaryDirectory(dir=args.output, prefix='.headers-') as staging:
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
                for member in archive.getmembers():
                    parts = Path(member.name).parts[1:]
                    if not parts:
                        continue
                    if Path(member.name).is_absolute() or '..' in parts or not (member.isfile() or member.isdir()):
                        raise ValueError(f'Invalid archive member: {member.name}')
                    member.name = str(Path(*parts))
                    archive.extract(member, staging)
            if not (Path(staging)/header).is_file():
                raise ValueError(f'{name} archive missing {header}')
            shutil.move(staging, destination)
        record = dict(name=name, version=version, url=url, sha256=actual, bytes=len(data))
        by_name[name] = record
        manifest.write_text(json.dumps(list(by_name.values()), indent=2)+'\n')


if __name__ == '__main__':
    main()
