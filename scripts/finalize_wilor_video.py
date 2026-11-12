#!/usr/bin/env python3
"""Install the legacy-skeleton/shared-mesh video at the standard final path."""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import uuid


def install_file(source, target):
    if target.exists() and os.path.samefile(source, target):
        return
    temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.tmp')
    try:
        try:
            os.link(source, temporary)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def write_metadata(path, metadata):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(metadata, indent=2) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def finalize(source, output, archive):
    source, output, archive = (Path(path).resolve() for path in (source, output, archive))
    if len({source, output, archive}) != 3:
        raise ValueError('source, final output and skeleton archive must be different paths')
    metadata = json.loads(source.with_suffix('.json').read_text())
    if metadata.get('mesh', {}).get('model') != 'WiLoR shared MANO':
        raise ValueError('expected the verified legacy-skeleton/shared-MANO-mesh video')
    audit_source = source.with_suffix('.mesh_projections.jsonl')
    if not source.is_file() or not audit_source.is_file():
        raise ValueError('video or mesh projection audit is missing')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        previous = json.loads(output.with_suffix('.json').read_text())
        if previous.get('mesh', {}).get('model') != 'WiLoR shared MANO':
            if archive.exists() and not os.path.samefile(output, archive):
                raise FileExistsError(f'preserve existing skeleton archive: {archive}')
            install_file(output, archive)
            previous['output'] = str(archive)
            write_metadata(archive.with_suffix('.json'), previous)
    audit_output = output.with_suffix('.mesh_projections.jsonl')
    install_file(audit_source, audit_output)
    install_file(source, output)
    metadata.update({'output': str(output), 'mesh_projections': str(audit_output),
                     'visualization_source_video': str(source)})
    if archive.is_file():
        metadata['original_skeleton_video'] = str(archive)
    write_metadata(output.with_suffix('.json'), metadata)
    print(f'WiLoR final video with matching shared mesh: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    finalize(args.source, args.output, args.archive)


if __name__ == '__main__':
    main()
