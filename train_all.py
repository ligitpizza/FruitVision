"""
train_all.py — runs every trainable model's train script from one command.

Usage:
    python train_all.py              # sequential (safe default)
    python train_all.py --parallel   # runs everything after the first entry concurrently

Place this file at the PROJECT ROOT (same level as member_apps/, core_modules/).

Covers:
  - The 4 original SVM ensemble members (member_1_ab .. member_4_da)
  - merged_1_4 (feature-level fusion of member 1 + member 4 into one SVM)
  - m14v2 (merged_1_4 plus texture -- colour+shape+gabor+texture, one SVM)
  - m14v3 (same features as m14v2, but detection combines member 1's Otsu
    box + member 4's HSV-saturation box via union, and calibration uses
    member 4's deskew)
  - yolo_pure (YOLOv8-cls fine-tuning, NOT an SVM -- needs
    datasets/yolo_cls/{fruit}/{train,val}/{ripe,unripe,rotten}/ to already
    exist; run pipeline/pure_yolo/dataset_prep.py first if it doesn't. If
    that dataset is missing, yolo_cls_train.py prints "Skipping <fruit>"
    per fruit and exits 0 -- it will show as [OK] here even though nothing
    actually got trained, so check its log if accuracy_summary.png doesn't
    show up under outputs/training/yolo_pure/ afterward.)
"""
import subprocess
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# (directory relative to PROJECT_ROOT, script to run inside it, tag used for
# the trained_logs/<tag>_train.log filename and the summary table)
MEMBERS = [
    ("member_apps/member_1_ab", "m1_train.py", "member_1_ab"),
    ("member_apps/member_2_bc", "m2_train.py", "member_2_bc"),
    ("member_apps/member_3_cd", "m3_train.py", "member_3_cd"),
    ("member_apps/member_4_da", "m4_train.py", "member_4_da"),
    ("member_apps/merged_member_1_4", "m14_train.py", "merged_member_1_4"),
    ("member_apps/merged_member_1_4_v2", "m14v2_train.py", "merged_member_1_4_v2"),
    ("member_apps/merged_member_1_4_v3", "m14v3_train.py", "merged_member_1_4_v3"),
    ("pipeline/pure_yolo", "yolo_cls_train.py", "yolo_pure"),
]


def run_one(rel_dir, script, tag):
    member_dir = PROJECT_ROOT / rel_dir
    log_path = PROJECT_ROOT / "trained_logs" / f"{tag}_train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            [sys.executable, script],
            cwd=member_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - start
    success = proc.returncode == 0
    status = "OK" if success else f"FAILED (exit {proc.returncode})"
    print(f"[{tag}] {status} — {elapsed:.1f}s — log: {log_path.name}")
    return tag, success, elapsed


def main():
    parser = argparse.ArgumentParser(description="Train every FruitVision model.")
    parser.add_argument("--parallel", action="store_true",
                         help="Run everything after the first entry concurrently.")
    args = parser.parse_args()

    print(f"Training {len(MEMBERS)} models "
          f"({'first entry then the rest in parallel' if args.parallel else 'sequentially'})...\n")

    results = []
    results.append(run_one(*MEMBERS[0]))

    remaining = MEMBERS[1:]
    if args.parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=len(remaining)) as pool:
            futures = [pool.submit(run_one, rel_dir, script, tag) for rel_dir, script, tag in remaining]
            for f in as_completed(futures):
                results.append(f.result())
    else:
        for rel_dir, script, tag in remaining:
            results.append(run_one(rel_dir, script, tag))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    failed = [r for r in results if not r[1]]
    for tag, success, elapsed in results:
        # ASCII keeps the summary printable in Windows consoles using cp1252.
        mark = "[OK]" if success else "[FAIL]"
        print(f"  {mark} {tag:20s} {elapsed:6.1f}s")

    total = sum(r[2] for r in results)
    print(f"\nTotal wall time (first entry sequential + rest): {total:.1f}s")

    if failed:
        print(f"\n{len(failed)} model(s) failed — check their .log files above.")
        sys.exit(1)
    else:
        print("\nAll models trained successfully.")


if __name__ == "__main__":
    main()
