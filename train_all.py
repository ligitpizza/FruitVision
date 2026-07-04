"""
train_all.py — runs every member's train.py from one command.

Usage:
    python train_all.py              # sequential (safe default)
    python train_all.py --parallel   # runs members 2-4 concurrently

Place this file at the PROJECT ROOT (same level as member_apps/, core_modules/).
"""
import subprocess
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

MEMBERS = [
    ("member_1_ab", "m1_train.py"),
    ("member_2_bc", "m2_train.py"),
    ("member_3_cd", "m3_train.py"),
    ("member_4_da", "m4_train.py"),
]


def run_one(folder, script):
    member_dir = PROJECT_ROOT / "member_apps" / folder
    log_path = PROJECT_ROOT / "trained_logs" / f"{folder}_train.log"
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
    print(f"[{folder}] {status} — {elapsed:.1f}s — log: {log_path.name}")
    return folder, success, elapsed


def main():
    parser = argparse.ArgumentParser(description="Train all FruitVision members.")
    parser.add_argument("--parallel", action="store_true",
                         help="Run members 2-4 concurrently after member 1 finishes.")
    args = parser.parse_args()

    print(f"Training {len(MEMBERS)} members "
          f"({'member 1 then 2-4 in parallel' if args.parallel else 'sequentially'})...\n")

    results = []
    results.append(run_one(*MEMBERS[0]))

    remaining = MEMBERS[1:]
    if args.parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=len(remaining)) as pool:
            futures = [pool.submit(run_one, folder, script) for folder, script in remaining]
            for f in as_completed(futures):
                results.append(f.result())
    else:
        for folder, script in remaining:
            results.append(run_one(folder, script))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    failed = [r for r in results if not r[1]]
    for folder, success, elapsed in results:
        mark = "✓" if success else "✗"
        print(f"  {mark} {folder:16s} {elapsed:6.1f}s")

    total = sum(r[2] for r in results)
    print(f"\nTotal wall time (member 1 sequential + rest): {total:.1f}s")

    if failed:
        print(f"\n{len(failed)} member(s) failed — check their .log files above.")
        sys.exit(1)
    else:
        print("\nAll members trained successfully.")


if __name__ == "__main__":
    main()
