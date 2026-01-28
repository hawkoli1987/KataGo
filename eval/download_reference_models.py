#!/usr/bin/env python3
"""Download historical KataGo models representing different strength levels.

This script downloads a curated set of KataGo models spanning the training history,
from early weak models to the strongest available. Models are saved with level prefixes
for use in the ladder evaluation pipeline.

Usage:
    python eval/download_reference_models.py --output-dir eval/assets/models
    python eval/download_reference_models.py --output-dir eval/assets/models --dry-run
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

# Curated list of KataGo models spanning different strength levels
# Format: (level, name, url, approx_elo, sha256)
# Models selected from https://katagotraining.org/networks/
# Note: Using .txt.gz extension for early models (they use text format)
REFERENCE_MODELS = [
    # Level 1: Very early training (~1000 Elo equivalent, b6c96 architecture)
    (1, "kata1-b6c96-s175395328-d26788732", 
     "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b6c96-s175395328-d26788732.txt.gz",
     1000, None),
    
    # Level 2: Early training (~1500 Elo, b10c128)
    (2, "kata1-b10c128-s197428736-d67404019",
     "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b10c128-s197428736-d67404019.txt.gz",
     1500, None),
    
    # Level 3: Mid-early training (~2000 Elo, b15c192)
    (3, "kata1-b15c192-s497233664-d149638345",
     "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b15c192-s497233664-d149638345.txt.gz",
     2000, None),
    
    # Level 4: Mid training (~2500 Elo, b20c256x2)
    (4, "kata1-b20c256x2-s668214784-d222255714",
     "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b20c256x2-s668214784-d222255714.txt.gz",
     2500, None),
    
    # Note: Level 5 skipped due to lack of publicly available ~3000 Elo model
    
    # Level 5: Late training (~3300 Elo, b40c256x2)
    (5, "kata1-b40c256x2-s5095420928-d1229425124",
     "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b40c256x2-s5095420928-d1229425124.bin.gz",
     3300, None),
    
    # Level 6: Very strong (~3500 Elo, b18c384nbt)
    (6, "kata1-b18c384nbt-s7709731328-d3715293823",
     "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s7709731328-d3715293823.bin.gz",
     3500, None),
    
    # Level 7: Near-final kata1 (~3700 Elo, b40c256 near-end)
    (7, "kata1-b40c256-s11840935168-d2898845681",
     "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b40c256-s11840935168-d2898845681.bin.gz",
     3700, None),
    
    # Level 8: kata1-b18c384 strong variant (~3800 Elo)
    (8, "kata1-b18c384nbt-s6582191360-d3422816034",
     "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s6582191360-d3422816034.bin.gz",
     3800, None),
    
    # Level 9: Strongest available (~4000+ Elo, b28c512)
    (9, "kata1-b28c512nbt-s12223529728-d5663671073",
     "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-s12223529728-d5663671073.bin.gz",
     4000, None),
]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_file(url: str, dest: Path, verbose: bool = True) -> bool:
    """Download a file from URL to destination path."""
    if verbose:
        print(f"  Downloading from {url}")
    
    try:
        # Add User-Agent header to avoid 403 errors
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; KataGo-Eval/1.0)"}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            with open(dest, "wb") as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"  ERROR: Failed to download: {e}")
        return False


def create_manifest_from_local(local_dir: Path, output_dir: Path):
    """Create manifest from local model files.
    
    Expects files named like: level_01_*.bin.gz or just *.bin.gz
    """
    import re
    
    models = []
    model_files = sorted(local_dir.glob("*.bin.gz"))
    
    for i, model_file in enumerate(model_files, 1):
        name = model_file.stem.replace(".bin", "")
        
        # Try to extract level from filename
        level_match = re.match(r"level_(\d+)_", model_file.name)
        if level_match:
            level = int(level_match.group(1))
        else:
            level = i
        
        # Estimate Elo based on level (rough approximation)
        approx_elo = 1000 + (level - 1) * 300
        
        sha256 = compute_sha256(model_file)
        
        models.append({
            "level": level,
            "name": name,
            "filename": model_file.name,
            "approx_elo": approx_elo,
            "url": None,
            "sha256": sha256
        })
        print(f"Found: Level {level}: {model_file.name} (~{approx_elo} Elo)")
    
    # Sort by level
    models.sort(key=lambda x: x["level"])
    
    # Write manifest
    manifest = {"models": models}
    manifest_path = output_dir / "manifest.json"
    
    # Copy files if local_dir != output_dir
    if local_dir.resolve() != output_dir.resolve():
        output_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for model in models:
            src = local_dir / model["filename"]
            dst = output_dir / model["filename"]
            if not dst.exists():
                print(f"Copying {src} -> {dst}")
                shutil.copy2(src, dst)
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nManifest written to: {manifest_path}")
    print(f"Total models: {len(models)}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Download KataGo reference models for ladder evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/models"),
        help="Directory to save models (default: assets/models)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated list of levels to download (e.g., '1,2,3'). Default: all"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file exists"
    )
    parser.add_argument(
        "--from-local",
        type=Path,
        default=None,
        help="Create manifest from local model directory instead of downloading"
    )
    args = parser.parse_args()
    
    # Handle local mode
    if args.from_local:
        if not args.from_local.exists():
            print(f"ERROR: Local directory not found: {args.from_local}")
            return
        create_manifest_from_local(args.from_local, args.output_dir)
        print("\nDone!")
        return
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse levels filter
    levels_filter = None
    if args.levels:
        levels_filter = set(int(x.strip()) for x in args.levels.split(","))
    
    # Track downloaded models for manifest
    manifest = {"models": []}
    
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Levels to download: {levels_filter if levels_filter else 'all'}")
    print()
    
    for level, name, url, approx_elo, expected_sha256 in REFERENCE_MODELS:
        if levels_filter and level not in levels_filter:
            continue
        
        # Determine output filename - use the extension from the URL
        url_ext = ".txt.gz" if url.endswith(".txt.gz") else ".bin.gz"
        filename = f"level_{level:02d}_{name}{url_ext}"
        filepath = output_dir / filename
        
        print(f"Level {level}: {name} (~{approx_elo} Elo)")
        
        if args.dry_run:
            print(f"  Would download to: {filepath}")
            print(f"  URL: {url}")
            manifest["models"].append({
                "level": level,
                "name": name,
                "filename": filename,
                "approx_elo": approx_elo,
                "url": url,
                "sha256": expected_sha256
            })
            continue
        
        # Check if file already exists
        if filepath.exists() and not args.force:
            print(f"  Already exists: {filepath}")
            sha256 = compute_sha256(filepath)
            manifest["models"].append({
                "level": level,
                "name": name,
                "filename": filename,
                "approx_elo": approx_elo,
                "url": url,
                "sha256": sha256
            })
            continue
        
        # Download the file
        success = download_file(url, filepath)
        if success:
            sha256 = compute_sha256(filepath)
            print(f"  Saved to: {filepath}")
            print(f"  SHA256: {sha256}")
            manifest["models"].append({
                "level": level,
                "name": name,
                "filename": filename,
                "approx_elo": approx_elo,
                "url": url,
                "sha256": sha256
            })
        else:
            print(f"  FAILED to download level {level}")
    
    # Write manifest
    manifest_path = output_dir / "manifest.json"
    if not args.dry_run:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest written to: {manifest_path}")
    else:
        print(f"\nWould write manifest to: {manifest_path}")
        print(json.dumps(manifest, indent=2))
    
    print("\nDone!")


if __name__ == "__main__":
    main()
