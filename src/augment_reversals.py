"""
Synthetic Reversal Augmentation Utility.
Generates mirrored and inverted character samples from the Normal handwriting dataset
to eliminate character-class confounding and teach the models universal reversal recognition.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageOps
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import TRAIN_DIR, TEST_DIR


# Characters that exhibit distinct visual reversal when horizontally mirrored
ASYMMETRIC_CHARACTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'J', 'K', 'L', 'N', 'P', 'Q', 'R', 'S', 'Z',
    'b', 'c', 'd', 'e', 'f', 'g', 'j', 'k', 'p', 'q', 'r', 's', 'z',
    '1', '2', '3', '4', '5', '6', '7', '9'
]


def generate_synthetic_reversals(
    train_dir: Path = TRAIN_DIR,
    samples_per_char: int = 500,
    vertical_flip_ratio: float = 0.25,
    dry_run: bool = False
) -> int:
    """
    Scans the Normal dataset folder for characters, generates horizontally
    (and optionally vertically) mirrored variants, and saves them to the Reversal folder.
    """
    normal_dir = train_dir / "Normal"
    reversal_dir = train_dir / "Reversal"

    if not normal_dir.exists():
        raise FileNotFoundError(f"Normal directory not found: {normal_dir}")
    reversal_dir.mkdir(parents=True, exist_ok=True)

    normal_files = [f for f in normal_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    print(f"[INFO] Found {len(normal_files)} normal training images.")

    # Group files by character prefix
    char_map = {}
    for f in normal_files:
        prefix = f.name.split('-')[0].split('_')[0]
        if prefix not in char_map:
            char_map[prefix] = []
        char_map[prefix].append(f)

    print(f"[INFO] Identified {len(char_map)} character groups.")
    generated_count = 0

    for char_key, file_list in char_map.items():
        # Target asymmetric characters or capital letters missing from Reversal
        target_files = file_list[:samples_per_char]
        for idx, src_file in enumerate(target_files):
            out_name = f"synthetic_rev_{char_key}_{idx}.png"
            out_path = reversal_dir / out_name

            if out_path.exists():
                continue

            if dry_run:
                generated_count += 1
                continue

            try:
                img = Image.open(src_file)
                # Apply horizontal mirror flip (classic dyslexia reversal)
                # Occasionally apply vertical flip for upside-down reversal
                if idx % int(1.0 / max(0.01, vertical_flip_ratio)) == 0:
                    flipped_img = ImageOps.flip(img)  # Vertical inversion (e.g. inverted A)
                else:
                    flipped_img = ImageOps.mirror(img)  # Horizontal mirror (e.g. b/d, reversed F)

                flipped_img.save(out_path)
                generated_count += 1
            except Exception as e:
                print(f"[WARNING] Could not process {src_file}: {e}")

    print(f"[SUCCESS] {'[DRY RUN] Would generate' if dry_run else 'Generated'} {generated_count} synthetic reversal images in {reversal_dir}.")
    return generated_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic reversal handwriting images.")
    parser.add_argument("--samples-per-char", type=int, default=300, help="Number of synthetic samples per character")
    parser.add_argument("--dry-run", action="store_true", help="Preview generation without writing files")
    args = parser.parse_args()

    generate_synthetic_reversals(samples_per_char=args.samples_per_char, dry_run=args.dry_run)
