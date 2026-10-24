"""Keep only PrimeVul functions that come from C source files.

Reads the six PrimeVul JSONL files from data/raw/primevul/ and writes filtered
copies with the same names to data/processed/primevul_c/. The raw files are
never modified.

A function is kept when its `file_name` ends in one of the allowed extensions
(default: `.c`). Rows with no file name (`"None"`, about a third of train,
mostly Chrome/C++ code) are dropped, since we can't tell their language.

The paired files store each pair as two consecutive lines (vulnerable first,
then patched). A pair is kept only if both functions pass the filter, so the
output stays a clean list of pairs.

Usage (from the repo root):
    python src/filter_primevul_c.py
    python src/filter_primevul_c.py --ext .c .h   # also keep header files
"""

import argparse
import json
from collections import Counter
from pathlib import Path

SPLITS = ["train", "valid", "test"]


def file_ext(row):
    name = row.get("file_name")
    if name in (None, "None", ""):
        return None
    return Path(name).suffix


def label(row):
    return "vuln" if row["target"] == 1 else "safe"


def filter_unpaired(src, dst, exts):
    counts = Counter()
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            row = json.loads(line)
            ext = file_ext(row)
            if ext in exts:
                fout.write(line)
                counts[f"kept_{label(row)}"] += 1
            else:
                counts[f"dropped_{label(row)}"] += 1
                counts["dropped_no_name" if ext is None else "dropped_other_ext"] += 1
    return counts


def filter_paired(src, dst, exts):
    counts = Counter()
    with open(src) as fin, open(dst, "w") as fout:
        lines = fin.readlines()
        assert len(lines) % 2 == 0, f"{src} has an odd number of lines"
        for vuln_line, safe_line in zip(lines[0::2], lines[1::2]):
            vuln, safe = json.loads(vuln_line), json.loads(safe_line)
            assert (vuln["target"], safe["target"]) == (1, 0), f"unexpected pair order at idx {vuln['idx']}"
            if file_ext(vuln) in exts and file_ext(safe) in exts:
                fout.write(vuln_line)
                fout.write(safe_line)
                counts["kept_pairs"] += 1
            else:
                counts["dropped_pairs"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/primevul"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/primevul_c"))
    parser.add_argument("--ext", nargs="+", default=[".c"], help="file extensions to keep (default: .c)")
    args = parser.parse_args()

    exts = set(args.ext)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Keeping functions from files ending in: {', '.join(sorted(exts))}\n")

    for split in SPLITS:
        name = f"primevul_{split}.jsonl"
        c = filter_unpaired(args.raw_dir / name, args.out_dir / name, exts)
        kept = c["kept_vuln"] + c["kept_safe"]
        dropped = c["dropped_vuln"] + c["dropped_safe"]
        print(
            f"{split:>5}: kept {kept:>6} (vuln {c['kept_vuln']:>5}, safe {c['kept_safe']:>6}) | "
            f"dropped {dropped:>6} (vuln {c['dropped_vuln']:>5}, safe {c['dropped_safe']:>6}; "
            f"no file name {c['dropped_no_name']}, other ext {c['dropped_other_ext']})"
        )

    print()
    for split in SPLITS:
        name = f"primevul_{split}_paired.jsonl"
        c = filter_paired(args.raw_dir / name, args.out_dir / name, exts)
        print(f"{split:>5} paired: kept {c['kept_pairs']:>5} pairs | dropped {c['dropped_pairs']:>5} pairs")

    print(f"\nWrote filtered files to {args.out_dir}/")


if __name__ == "__main__":
    main()
