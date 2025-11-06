#!/usr/bin/env python3
"""
Rename image files in a directory to sequential zero-padded numbers.

Examples:
  # Dry-run, show mapping
  python scripts/rename_images.py data/images/train --dry-run

  # Rename starting at 0 with 3-digit padding
  python scripts/rename_images.py data/images/train --start 0 --padding 3

  # Rename starting at 100 with 4-digit padding, sort by mtime
  python scripts/rename_images.py data/images/train --start 100 --padding 4 --sort mtime

This script is safe: it computes the final mapping, checks for conflicts, then performs a two-step rename (temp names) to avoid collisions.
"""

import argparse
from pathlib import Path
from uuid import uuid4
import sys

DEFAULT_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']


def parse_args():
    p = argparse.ArgumentParser(description='Rename images in a folder to sequential zero-padded numbers')
    p.add_argument('folder', type=Path, help='Folder containing image files to rename')
    p.add_argument('--start', type=int, default=0, help='Start index (default: 0)')
    p.add_argument('--padding', type=int, default=3, help='Zero padding width (default: 3 -> 000)')
    p.add_argument('--sort', choices=['name', 'mtime'], default='name', help='Sort files by name or mtime (default: name)')
    p.add_argument('--dry-run', action='store_true', help='Show planned renames but do not perform them')
    p.add_argument('--exts', type=str, default=','.join(DEFAULT_EXTS),
                   help='Comma-separated list of extensions to include (default: %s)' % ','.join(DEFAULT_EXTS))
    p.add_argument('--force', action='store_true', help='Force rename even if padded capacity is exceeded')
    return p.parse_args()


def collect_files(folder: Path, exts, sort='name'):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if sort == 'name':
        files.sort(key=lambda p: p.name.lower())
    else:
        files.sort(key=lambda p: p.stat().st_mtime)
    return files


def main():
    args = parse_args()
    folder: Path = args.folder
    if not folder.exists():
        print(f'Error: folder "{folder}" does not exist', file=sys.stderr)
        sys.exit(2)
    if not folder.is_dir():
        print(f'Error: "{folder}" is not a directory', file=sys.stderr)
        sys.exit(2)

    exts = [e.lower() if e.startswith('.') else ('.' + e.lower()) for e in [x.strip() for x in args.exts.split(',') if x.strip()]]
    files = collect_files(folder, exts, sort=args.sort)

    if not files:
        print('No files found for the given extensions:', exts)
        return

    total = args.start + len(files) - 1
    capacity = 10 ** args.padding - 1
    if total > capacity and not args.force:
        print(f'ERROR: padding {args.padding} cannot represent index {total} (max {capacity}).', file=sys.stderr)
        print('Either increase --padding or use --force to proceed (risking non-zero padding).', file=sys.stderr)
        sys.exit(3)

    # Build mapping
    mapping = {}
    targets = set()
    for i, p in enumerate(files, start=args.start):
        new_name = f"{i:0{args.padding}d}" + p.suffix.lower()
        target = folder / new_name
        mapping[p] = target
        targets.add(target)

    # Check for conflicts with files not in source list
    existing_non_sources = [p for p in folder.iterdir() if p.is_file() and p not in files and (folder / p.name) in targets]
    if existing_non_sources:
        print('ERROR: Found existing files that would be overwritten and are not part of the rename set:', file=sys.stderr)
        for p in existing_non_sources:
            print('  ', p, file=sys.stderr)
        print('Move or remove these files first, or adjust your settings.', file=sys.stderr)
        sys.exit(4)

    # Dry-run prints mapping
    if args.dry_run:
        print('Dry-run: planned renames:')
        for old, new in mapping.items():
            print(f'  {old.name} -> {new.name}')
        print(f'Total files: {len(files)}')
        return

    # If any target file already exists but is also in the source list, we'll avoid collision via temp names
    uid = uuid4().hex[:8]
    temp_map = {}
    try:
        # Step 1: rename all sources to temp names
        for idx, (old, final) in enumerate(mapping.items()):
            temp_name = f'.tmp_ren_{uid}_{idx}{old.suffix.lower()}'
            temp_path = folder / temp_name
            # If temp exists unexpectedly, choose another
            while temp_path.exists():
                temp_name = f'.tmp_ren_{uuid4().hex[:8]}_{idx}{old.suffix.lower()}'
                temp_path = folder / temp_name
            old.rename(temp_path)
            temp_map[temp_path] = final

        # Step 2: rename temps to final names
        for temp_path, final in temp_map.items():
            temp_path.rename(final)

        print(f'Renamed {len(files)} files in "{folder}" starting at {args.start} with padding {args.padding}.')
    except Exception as e:
        print('ERROR during rename:', e, file=sys.stderr)
        print('Attempting to rollback partial renames...', file=sys.stderr)
        # Attempt best-effort rollback: move any temp files back to original names if possible
        for temp, final in temp_map.items():
            try:
                if temp.exists():
                    # recover original name by extracting index from temp mapping
                    # Not always possible; skip gracefully
                    temp.rename(folder / ('recovered_' + temp.name))
            except Exception:
                pass
        sys.exit(5)


if __name__ == '__main__':
    main()
