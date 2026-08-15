from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/PST50"),
                        help="Root of the raw PST50 dataset (default: data/PST50)")
    parser.add_argument("--out", type=Path, default=Path("data/PST50.txt"),
                        help="Output TXT path (default: data/PST50.txt)")
    parser.add_argument("--content-src", choices=["content_log", "content_709"],
                        default="content_log",
                        help="Which subdir to use as content (default: content_log)")
    parser.add_argument("--num", type=int, default=50, help="Number of samples (default: 50)")
    # Kept as a no-op for compatibility with earlier invocations. Paths are now
    # always relative to the output list so the project can be moved freely.
    parser.add_argument("--relative", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = args.root
    out = args.out
    content_dir = root / args.content_src
    ref_dir = root / "paired_style"
    gt_dir = root / "paired_gt"

    for d in (content_dir, ref_dir, gt_dir):
        if not d.is_dir():
            raise SystemExit(f"[error] required directory not found: {d}")

    def fmt(p: Path) -> str:
        return Path(os.path.relpath(p, start=out.parent)).as_posix()

    rows, missing = [], []
    for i in range(1, args.num + 1):
        c = content_dir / f"in{i}.png"
        r = ref_dir / f"tar{i}.png"
        g = gt_dir / f"gt{i}.png"
        if not (c.exists() and r.exists() and g.exists()):
            missing.append(i)
            continue
        rows.append(f"{fmt(c)} {fmt(r)} {fmt(g)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + ("\n" if rows else ""))

    print(f"[info] root       : {root}")
    print(f"[info] content_src: {args.content_src}")
    print(f"[info] wrote      : {out}  ({len(rows)} rows)")
    if missing:
        print(f"[warn] {len(missing)} samples skipped (missing files): {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    main()
