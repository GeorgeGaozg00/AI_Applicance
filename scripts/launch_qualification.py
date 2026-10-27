#!/usr/bin/env python3
"""Launch a bounded qualification job under systemd, independent of SSH sessions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--reports', required=True)
    parser.add_argument('--log', required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error('Run with sudo: systemd and the NPU require administrator access')
    root = Path(__file__).resolve().parents[1]
    reports, log = Path(args.reports).resolve(), Path(args.log).resolve()
    if (reports / 'qualification.json').exists():
        parser.error('Use a fresh reports directory to preserve previous test evidence')
    model = Path(args.model).resolve(strict=True)
    reports.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    unit = 'ai-appliance-qualify-' + uuid.uuid4().hex[:12]
    command = ['systemd-run', '--unit=' + unit, '--collect', '--property=Type=exec',
               '--property=WorkingDirectory=' + str(root),
               '--property=RuntimeMaxSec=2400',
               '--property=StandardOutput=append:' + str(log),
               '--property=StandardError=append:' + str(log), sys.executable,
               str(root / 'scripts' / 'qualify.py'), '--image', args.image,
               '--model', str(model), '--reports', str(reports)]
    subprocess.run(command, check=True)
    print(json.dumps({'unit': unit + '.service', 'report': str(reports / 'qualification.json'),
                      'log': str(log)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
