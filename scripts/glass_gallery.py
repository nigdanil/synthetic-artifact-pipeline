"""Generate a local, offline visual QA gallery from 100 verified glass previews."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sapipeline.batch.glass_gallery import generate_glass_gallery
from sapipeline.batch.runner import BatchRunError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build'])
    args = parser.parse_args()
    try:
        if args.command == 'build':
            path, count = generate_glass_gallery(ROOT)
            print(f'PRIVATE GLASS GALLERY: {path} :: {count} QA PASS images / {count // 2} scenes')
            print('Manual review decisions require an explicit JSON download. No training approval.')
    except BatchRunError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
