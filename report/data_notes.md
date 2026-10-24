# Data notes

Working notes on data preparation, to feed into the final report.

## Source

PrimeVul (https://github.com/DLVulDet/PrimeVul), downloaded via the links in its
README into `data/raw/primevul/`: `primevul_{train,valid,test}.jsonl` and the
paired versions `primevul_{train,valid,test}_paired.jsonl`. Each row is one
function with `func` (source code), `target` (1 = vulnerable, 0 = safe), `cwe`,
`project`, `commit_id`, `file_name` and a few other metadata fields.

## Step 1: keep only C functions

Script: `src/filter_primevul_c.py`. It writes to `data/processed/primevul_c/`
(same file names) and leaves the raw data untouched.

Rule: keep a function if its `file_name` ends in `.c`.

- **Rows without a file name are dropped.** About 31% of train has
  `file_name = "None"` (mostly Chrome, which is C++). We can't tell their
  language, so we leave them out rather than guess. `file_info.json` holds the
  same names and doesn't cover the unnamed rows either, so it isn't used.
- **Header files (`.h`) are excluded for now.** They'd add about 5k train rows,
  but headers are often shared with C++ code. `--ext .c .h` adds them back if we
  change this.
- **Pairs stay intact.** In the paired files, each pair is two consecutive lines
  (vulnerable, then patched). A pair is kept only if both functions are from
  `.c` files. 19 pairs have different commit ids for the two halves (the fix
  landed in a fork or a later commit), but they are still the same function, so
  they're kept like any other pair.

### Counts

| Split | Raw rows | Kept | Kept vulnerable | Kept safe | % vulnerable | Dropped: no file name | Dropped: other extension |
|---|---|---|---|---|---|---|---|
| train | 175,797 | 102,564 | 2,993 | 99,571 | 2.9% | 55,260 | 17,973 |
| valid | 23,948 | 15,752 | 362 | 15,390 | 2.3% | 4,299 | 3,897 |
| test | 24,788 | 15,073 | 341 | 14,732 | 2.3% | 5,910 | 3,805 |

| Paired split | Raw pairs | Kept pairs | Dropped pairs |
|---|---|---|---|
| train | 3,789 | 2,062 | 1,727 |
| valid | 480 | 292 | 188 |
| test | 435 | 253 | 182 |

Other extensions dropped: mainly `.cpp`, `.cc` and `.h`.

### Takeaway

The C data is heavily imbalanced: only about 3% of functions are vulnerable. A
model that always answers "SAFE" would score about 97% accuracy, so accuracy on
its own is meaningless here. Training needs a balanced subset (see `plan.md`),
and evaluation should report precision, recall, F1 and paired accuracy.
