#!/usr/bin/env python3
"""
Repository hygiene guard for tracked images and binary assets.
Standard library only.

Fails if:
1. Any tracked file has a forbidden extension: *.pth, *.npz, *.tif, *.tiff, *.zip
2. Any tracked image is outside allowed directories:
   - dashboard/public/data/
   - reports/real/figures/
   - dashboard/docs/screenshots/
3. Any single tracked image exceeds 500 KB (500 * 1024 bytes).
4. Total size of all tracked images exceeds 20 MB (20 * 1024 * 1024 bytes).
"""

import os
import sys
import subprocess
from pathlib import Path

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.svg', '.ico', '.bmp', '.tif', '.tiff'}
FORBIDDEN_EXTENSIONS = {'.pth', '.npz', '.tif', '.tiff', '.zip'}

ALLOWED_DIRECTORIES = [
    'dashboard/public/data/',
    'reports/real/figures/',
    'dashboard/docs/screenshots/'
]

MAX_SINGLE_IMAGE_BYTES = 500 * 1024       # 500 KB
MAX_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


def main():
    try:
        output = subprocess.check_output(['git', 'ls-files'], text=True)
    except Exception as e:
        print(f"ERROR: Failed to run 'git ls-files': {e}", file=sys.stderr)
        sys.exit(1)

    tracked_files = [line.strip().replace('\\', '/') for line in output.strip().splitlines() if line.strip()]

    errors = []

    # 1. Check for forbidden binary extensions
    forbidden_files = []
    for f in tracked_files:
        suffix = Path(f).suffix.lower()
        if suffix in FORBIDDEN_EXTENSIONS:
            forbidden_files.append(f)
    if forbidden_files:
        errors.append(
            f"Found {len(forbidden_files)} tracked file(s) with forbidden extensions ({FORBIDDEN_EXTENSIONS}):\n"
            + "\n".join(f"  - {f}" for f in forbidden_files)
        )

    # 2. Check tracked images
    tracked_images = []
    for f in tracked_files:
        suffix = Path(f).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            tracked_images.append(f)

    # Directory constraint & single file size limit
    outside_dir = []
    oversized = []
    total_image_bytes = 0

    for img in tracked_images:
        # Normalize path
        norm_img = img.replace('\\', '/')
        
        # Check allowed directory
        is_allowed_dir = any(norm_img.startswith(allowed) for allowed in ALLOWED_DIRECTORIES)
        if not is_allowed_dir:
            outside_dir.append(norm_img)

        # Check file size on disk
        if os.path.exists(norm_img):
            sz = os.path.getsize(norm_img)
            total_image_bytes += sz
            if sz > MAX_SINGLE_IMAGE_BYTES:
                oversized.append((norm_img, sz))
        else:
            errors.append(f"Tracked image not found on disk: {norm_img}")

    if outside_dir:
        errors.append(
            f"Found {len(outside_dir)} tracked image(s) outside allowed directories ({ALLOWED_DIRECTORIES}):\n"
            + "\n".join(f"  - {f}" for f in outside_dir)
        )

    if oversized:
        errors.append(
            f"Found {len(oversized)} tracked image(s) exceeding {MAX_SINGLE_IMAGE_BYTES / 1024:.0f} KB limit:\n"
            + "\n".join(f"  - {f} ({sz / 1024:.1f} KB)" for f, sz in oversized)
        )

    if total_image_bytes > MAX_TOTAL_IMAGE_BYTES:
        errors.append(
            f"Total tracked image size {total_image_bytes / (1024 * 1024):.2f} MB exceeds {MAX_TOTAL_IMAGE_BYTES / (1024 * 1024):.0f} MB limit."
        )

    if errors:
        print("=" * 60, file=sys.stderr)
        print("CHECK_REPO_IMAGES GUARD FAILED:", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for err in errors:
            print(f"\n{err}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("CHECK_REPO_IMAGES GUARD PASSED")
    print("=" * 60)
    print(f"Tracked images checked: {len(tracked_images)}")
    print(f"Total image payload:    {total_image_bytes / 1024:.1f} KB ({total_image_bytes / (1024 * 1024):.2f} MB)")
    print(f"Forbidden files found:  0")
    print(f"Max file size observed: {max((os.path.getsize(f) for f in tracked_images), default=0) / 1024:.1f} KB (limit: 500 KB)")
    sys.exit(0)


if __name__ == '__main__':
    main()
