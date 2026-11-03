from collections import Counter
from pathlib import Path
import sys

import numpy as np

DATA_DIR = Path("/workspace/DataSet_00000_00999/_Merged_Data")
COLUMNS = ["index", "Temp1", "Press1", "Temp2", "Press2", "Acc", "Gas"]
EXPECTED_FILES = 1000
EXPECTED_SHAPE = (3000, 7)

if not DATA_DIR.is_dir():
    raise SystemExit(f"데이터 폴더를 찾지 못했습니다: {DATA_DIR}")

files = sorted(
    p for p in DATA_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() == ".csv"
)

print("Data directory:", DATA_DIR)
print("CSV files:", len(files))
print("Column order (provided specification):", COLUMNS)

if not files:
    raise SystemExit("CSV 파일이 없습니다.")

print("First CSV:", files[0].relative_to(DATA_DIR))
with files[0].open("r", encoding="utf-8-sig") as f:
    print("First raw line:", repr(f.readline().rstrip("\r\n")))

shapes = Counter()
index_steps = Counter()
zero_counts = np.zeros(7, dtype=np.int64)
errors = []
index_warnings = []
valid_files = 0
valid_rows = 0
total_bytes = sum(p.stat().st_size for p in files)

for number, path in enumerate(files, start=1):
    name = str(path.relative_to(DATA_DIR))

    try:
        # 헤더 없는 숫자 CSV: 첫 행을 건너뛰지 않는다.
        # pandas를 사용한다면 read_csv(..., header=None)에 해당한다.
        data = np.loadtxt(
            path,
            delimiter=",",
            dtype=np.float64,
            encoding="utf-8-sig",
            ndmin=2,
        )
    except Exception as exc:
        errors.append(f"{name}: 읽기 실패: {exc}")
        continue

    shapes[data.shape] += 1

    if data.shape != EXPECTED_SHAPE:
        errors.append(f"{name}: 예상과 다른 shape {data.shape}")
        continue

    if not np.isfinite(data).all():
        counts = (~np.isfinite(data)).sum(axis=0)
        errors.append(
            f"{name}: NaN/Inf 개수 {dict(zip(COLUMNS, counts.tolist()))}"
        )
        continue

    valid_files += 1
    valid_rows += data.shape[0]
    zero_counts += np.count_nonzero(data == 0, axis=0)

    differences = np.diff(data[:, 0])
    increasing = bool(np.all(differences > 0))
    uniform = bool(np.allclose(differences, differences[0]))

    if not increasing or not uniform:
        index_warnings.append(
            f"{name}: increasing={increasing}, uniform={uniform}"
        )

    index_steps[
        (float(differences.min()), float(differences.max()))
    ] += 1

    if valid_files == 1:
        print("\nFirst structurally valid CSV:", name)
        print("Shape:", data.shape)
        print("First 3 rows:\n", data[:3])
        print("Last row:\n", data[-1])
        print("Index start/end:", data[0, 0], data[-1, 0])

    if number % 100 == 0:
        print(f"Progress: {number}/{len(files)}")

print("\n--- Inspection result ---")
print("CSV total size (MiB):", round(total_bytes / 1024**2, 2))
print("Shape counts:", dict(shapes))
print("Valid files:", valid_files)
print("Valid rows:", valid_rows)
print("Zero counts:", dict(zip(COLUMNS, zero_counts.tolist())))
print("Index step (min, max) counts:", dict(index_steps))
print("Index warning count:", len(index_warnings))
print("Error count:", len(errors))

for message in index_warnings[:20]:
    print("INDEX WARNING:", message)

for message in errors[:20]:
    print("ERROR:", message)

if len(files) != EXPECTED_FILES:
    print(f"FILE COUNT ERROR: expected={EXPECTED_FILES}, actual={len(files)}")

if errors or len(files) != EXPECTED_FILES:
    sys.exit(1)

print("Structural checks passed.")
if index_warnings:
    print("Index warnings require review.")
