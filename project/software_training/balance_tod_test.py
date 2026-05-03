"""
balance_tod_test.py
===================
Moves symlinks from train/ to test/ in data_20k_day and data_20k_night
so that each test set is balanced (blank ≈ non_blank), excluding blacklisted images.

Run once before training. Safe to re-run — skips if already balanced.
"""

import random
from pathlib import Path

_PROJECT = Path(__file__).parent.parent
_BLACKLIST = set(
    l.strip() for l in (_PROJECT / "blacklist.txt").read_text().splitlines()
    if l.strip() and not l.startswith("#")
)

random.seed(99)


def balance(tod: str):
    root = _PROJECT / f"data_20k_{tod}"

    counts = {
        (sp, cls): list((root / sp / cls).iterdir())
        for sp in ("train", "test")
        for cls in ("blank", "non_blank")
    }

    n_blank     = len(counts[("test", "blank")])
    n_nonblank  = len(counts[("test", "non_blank")])
    target      = max(n_blank, n_nonblank)
    print(f"\n{tod.upper()}  test: {n_blank} blank / {n_nonblank} non_blank  → target {target} each")

    for cls, current_n in (("blank", n_blank), ("non_blank", n_nonblank)):
        deficit = target - current_n
        if deficit <= 0:
            print(f"  {cls}: already balanced")
            continue

        candidates = [
            p for p in counts[("train", cls)]
            if p.stem not in _BLACKLIST
        ]
        random.shuffle(candidates)
        chosen = candidates[:deficit]

        dst_dir = root / "test" / cls
        for src in chosen:
            dst = dst_dir / src.name
            src.rename(dst)

        print(f"  {cls}: moved {len(chosen)} from train → test  "
              f"(train now {len(counts[('train', cls)]) - len(chosen):,}, "
              f"test now {current_n + len(chosen):,})")


if __name__ == "__main__":
    balance("day")
    balance("night")
    print("\nDone. Final counts:")
    for tod in ("day", "night"):
        root = Path(__file__).parent.parent / f"data_20k_{tod}"
        print(f"  {tod.upper()}:")
        for sp in ("train", "test"):
            for cls in ("blank", "non_blank"):
                print(f"    {sp}/{cls}: {len(list((root/sp/cls).iterdir())):,}")
