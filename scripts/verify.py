#!/usr/bin/env python3
"""Read-only verification of the GitHub delivery and reported scientific results."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from verify_delivery import verify

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-publication', action='store_true',
                        help='Also check private article files when available locally')
    args = parser.parse_args()
    print(json.dumps(verify(include_publication=args.include_publication), indent=2))
