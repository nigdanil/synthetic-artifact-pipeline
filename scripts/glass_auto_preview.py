from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sapipeline.batch.glass_auto_preview import generate_auto_glass_previews
from sapipeline.batch.runner import BatchRunError


def main() -> int:
    parser = argparse.ArgumentParser(description='Automatic bbox-driven glass preview generator')
    sub = parser.add_subparsers(dest='cmd', required=True)
    prev = sub.add_parser('preview')
    prev.add_argument('--start', type=int, default=0)
    prev.add_argument('--count', type=int, default=5)
    args = parser.parse_args()
    try:
        if args.cmd == 'preview':
            outputs = generate_auto_glass_previews(ROOT, start=args.start, count=args.count)
            for folder, report in outputs:
                statuses = ', '.join(f"{row['variant']}={row['technical_qa']['status']}" for row in report['variants'])
                print(f'AUTO GLASS PREVIEW: {folder} :: {statuses}')
            print('No Stage-7 export, no training approval, source ZIP/SEG unchanged.')
    except BatchRunError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
