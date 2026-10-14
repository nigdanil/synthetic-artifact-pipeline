"""Private resumable 100-image bbox glass glare pilot (QA preview only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sapipeline.batch.glass_batch100 import pack_batch, run_batch
from sapipeline.batch.runner import BatchRunError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command_run = commands.add_parser('run', help='Resume generation, stop at 100 QA PASS images')
    command_run.add_argument('--max-new-scenes', type=int, default=60)
    commands.add_parser('pack', help='Create a private 100-image QA preview ZIP')
    args = parser.parse_args()
    try:
        if args.command == 'run':
            status = run_batch(ROOT, max_new_scenes=args.max_new_scenes)
            print('GLASS BATCH:', ' '.join(f'{key}={value}' for key, value in status.items()))
            if status['qa_pass'] != status['target']:
                print('INCOMPLETE: run again or review rejected scenes before pack.')
        else:
            path, count = pack_batch(ROOT)
            print(f'PRIVATE QA-PREVIEW ZIP: {count} images + SEG + mask visualization; {path}')
            print('Manual review required. No Stage-7 training approval.')
    except BatchRunError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
