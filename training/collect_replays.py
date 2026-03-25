"""
collect_replays.py
==================
CLI script: download and parse Rocket League replays from ballchasing.com.

Usage
-----
    python training/collect_replays.py \\
        --api-key YOUR_KEY \\
        --count 1000 \\
        --min-rank bronze-1 \\
        --max-rank bronze-3 \\
        --output-dir training/replay_data/

    # resume a previous run (skips already-downloaded replays):
    python training/collect_replays.py --api-key KEY --count 1000 --resume

    # keep raw .replay files (useful for debugging the parser):
    python training/collect_replays.py --api-key KEY --count 10 --keep-raw

API key
-------
Sign up at https://ballchasing.com and copy your API key from your profile page.
The key is passed via the Authorization header; do not commit it to source control.
Pass it via --api-key or set the BALLCHASING_API_KEY environment variable.

Rank names
----------
Valid values for --min-rank / --max-rank (low to high):
  unranked
  bronze-1 bronze-2 bronze-3
  silver-1 silver-2 silver-3
  gold-1 gold-2 gold-3
  platinum-1 platinum-2 platinum-3
  diamond-1 diamond-2 diamond-3
  champion-1 champion-2 champion-3
  grand-champion-1 grand-champion-2 grand-champion-3
  supersonic-legend
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from replay_sampler import collect, VALID_RANKS


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Download and parse Rocket League replays from ballchasing.com.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--api-key',
        default=os.environ.get('BALLCHASING_API_KEY', ''),
        help='ballchasing.com API key (or set BALLCHASING_API_KEY env var)',
    )
    parser.add_argument(
        '--count',
        type=int,
        default=1000,
        help='Number of replays to collect (default: 1000)',
    )
    parser.add_argument(
        '--min-rank',
        default='bronze-1',
        choices=VALID_RANKS,
        metavar='RANK',
        help=f'Minimum rank filter. One of: {", ".join(VALID_RANKS)} (default: bronze-1)',
    )
    parser.add_argument(
        '--max-rank',
        default='bronze-3',
        choices=VALID_RANKS,
        metavar='RANK',
        help='Maximum rank filter (default: bronze-3)',
    )
    parser.add_argument(
        '--playlist',
        default='ranked-standard',
        help='Playlist filter passed to ballchasing API (default: ranked-standard)',
    )
    parser.add_argument(
        '--output-dir',
        default='training/replay_data',
        help='Root output directory (default: training/replay_data)',
    )
    parser.add_argument(
        '--keep-raw',
        action='store_true',
        help='Keep raw .replay files after parsing (deleted by default)',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Skip replay IDs already present in manifest.json (default: on)',
    )
    parser.add_argument(
        '--no-resume',
        dest='resume',
        action='store_false',
        help='Re-download all replays even if already in manifest.json',
    )

    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            'No API key provided. Use --api-key or set BALLCHASING_API_KEY.'
        )

    collect(
        api_key    = args.api_key,
        count      = args.count,
        output_dir = Path(args.output_dir),
        min_rank   = args.min_rank,
        max_rank   = args.max_rank,
        playlist   = args.playlist,
        keep_raw   = args.keep_raw,
        resume     = args.resume,
    )


if __name__ == '__main__':
    main()
