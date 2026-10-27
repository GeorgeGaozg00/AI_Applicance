#!/usr/bin/env python3
"""Build a small source release; never package credentials, models or diagnostics."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.1.0'


def main():
    paths = []
    for item in ['appliance', 'scripts', 'tests', 'profiles', 'requirements', 'docs', 'pyproject.toml', 'README.md']:
        path = ROOT / item
        paths.extend([path] if path.is_file() else sorted(p for p in path.rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'))
    manifest = {'version': VERSION, 'files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in paths}}
    folder = ROOT / 'dist'
    folder.mkdir(exist_ok=True)
    dest = folder / f'ai-appliance-{VERSION}.tar.gz'
    with tarfile.open(dest, 'w:gz') as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('release-manifest.json')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    checksum = hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_suffix(dest.suffix + '.sha256').write_text(f'{checksum}  {dest.name}\n')
    print(json.dumps({'release': str(dest), 'sha256': checksum, 'files': len(paths)}))


if __name__ == '__main__':
    main()
