"""Read, validate, and save the training and evaluation grids separately."""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT = Path(__file__).resolve().parent
GRID_SHAPE = (72, 49)
HOURS = tuple(range(0, 24, 2))
TRAIN_START = date(2021, 1, 1)
TRAIN_END = date(2023, 12, 31)
EVAL_START = date(2024, 1, 1)
DEFAULT_EVAL_END = date(2024, 5, 31)
NAME = re.compile(r"([0-9]{8})_m_3039_H_T_([0-9]{2})([0-9]{2})\.csv")
TIMESTAMP_DTYPE = np.dtype("datetime64[m]")


def index_csv_files(source):
    paths = {}
    if not source.is_dir():
        raise ValueError(f"원본 폴더가 없습니다. {source}")
    for path in sorted(source.rglob("*")):
        if path.suffix.lower() != ".csv" or not path.is_file():
            continue
        match = NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"파일명 형식 오류. {path}")
        raw_date, raw_start, raw_end = match.groups()
        try:
            day = date.fromisoformat(raw_date)
        except ValueError as error:
            raise ValueError(f"파일명 날짜 오류. {path}") from error
        start, end = int(raw_start), int(raw_end)
        if start not in HOURS or end != start + 1:
            raise ValueError(f"파일명 시간대 오류. {path}")
        stamp = datetime.combine(day, time(start))
        if stamp in paths:
            raise ValueError(f"중복 시각 {stamp.isoformat()}. {paths[stamp]} / {path}")
        paths[stamp] = path
    if not paths:
        raise ValueError(f"CSV 파일이 없습니다. {source}")
    return dict(sorted(paths.items()))


def read_grid(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [line.rstrip("\r\n").split("|") for line in stream]
    if len(rows) != GRID_SHAPE[0]:
        raise ValueError(f"{path} 행 수 {len(rows)}, 예상 {GRID_SHAPE[0]}")
    for row_index, cells in enumerate(rows):
        if len(cells) != GRID_SHAPE[1]:
            raise ValueError(f"{path} 행 {row_index} 열 수 {len(cells)}, 예상 {GRID_SHAPE[1]}")
    try:
        grid = np.array(rows, dtype=np.float64)
    except (ValueError, OverflowError) as error:
        for row_index, row in enumerate(rows):
            for col_index, cell in enumerate(row):
                if not cell.strip():
                    raise ValueError(f"{path} 빈 셀 ({row_index}, {col_index})") from error
                try:
                    float(cell)
                except (ValueError, OverflowError):
                    raise ValueError(
                        f"{path} 숫자 변환 오류 ({row_index}, {col_index}) {cell!r}"
                    ) from error
        raise ValueError(f"{path} 숫자 변환 오류") from error
    finite = np.isfinite(grid)
    if not finite.all():
        row_index, col_index = np.argwhere(~finite)[0]
        raise ValueError(f"{path} NaN 또는 무한값 ({row_index}, {col_index})")
    return grid


def describe_period(paths, start, end):
    number_of_days = (end - start).days + 1
    expected = [
        datetime.combine(start + timedelta(days=day), time(hour))
        for day in range(number_of_days)
        for hour in HOURS
    ]
    observed = {stamp: paths[stamp] for stamp in expected if stamp in paths}
    missing = [stamp for stamp in expected if stamp not in observed]
    slots_by_day = Counter(stamp.date() for stamp in observed)
    report = {
        "requested_start": expected[0].isoformat(timespec="minutes"),
        "requested_end": expected[-1].isoformat(timespec="minutes"),
        "observed_start": min(observed).isoformat(timespec="minutes") if observed else None,
        "observed_end": max(observed).isoformat(timespec="minutes") if observed else None,
        "expected_slot_count": len(expected),
        "observed_slot_count": len(observed),
        "missing_slot_count": len(missing),
        "missing_timestamps": [stamp.isoformat(timespec="minutes") for stamp in missing],
        "duplicate_timestamp_count": 0,
        "requested_day_count": number_of_days,
        "observed_day_count": len(slots_by_day),
        "days_with_all_12_slots": sum(count == len(HOURS) for count in slots_by_day.values()),
        "complete_coverage": not missing,
    }
    return observed, report


def save_split(directory, name, paths):
    value_path = directory / f"{name}_values.npy"
    timestamp_path = directory / f"{name}_timestamps.npy"
    stamps = np.array(list(paths), dtype=TIMESTAMP_DTYPE)
    shape = (len(paths), *GRID_SHAPE)
    parsed_hash = hashlib.sha256()
    zero_count = 0
    negative_count = 0
    if paths:
        values = np.lib.format.open_memmap(value_path, mode="w+", dtype=np.float64, shape=shape)
        for index, path in enumerate(paths.values()):
            grid = read_grid(path)
            values[index] = grid
            parsed_hash.update(grid)
            zero_count += np.count_nonzero(grid == 0)
            negative_count += np.count_nonzero(grid < 0)
        values.flush()
        del values
    else:
        np.save(value_path, np.empty(shape, dtype=np.float64), allow_pickle=False)
    np.save(timestamp_path, stamps, allow_pickle=False)

    restored_values = np.load(value_path, mmap_mode="r", allow_pickle=False)
    restored_stamps = np.load(timestamp_path, allow_pickle=False)
    restored_hash = hashlib.sha256()
    for start in range(0, len(restored_values), 128):
        restored_hash.update(restored_values[start : start + 128])
    if (
        restored_values.shape != shape
        or restored_values.dtype != np.dtype(np.float64)
        or restored_stamps.dtype != TIMESTAMP_DTYPE
        or restored_stamps.shape != (restored_values.shape[0],)
        or not np.array_equal(restored_stamps, stamps)
        or np.any(restored_stamps[1:] <= restored_stamps[:-1])
        or restored_hash.digest() != parsed_hash.digest()
    ):
        raise ValueError(f"저장 후 재검증 실패. {name}")
    return {
        "values_file": value_path.name,
        "timestamps_file": timestamp_path.name,
        "shape": list(restored_values.shape),
        "dtype": str(restored_values.dtype),
        "timestamp_count": int(restored_stamps.size),
        "timestamp_dtype": str(restored_stamps.dtype),
        "first_saved_timestamp": str(restored_stamps[0]) if stamps.size else None,
        "last_saved_timestamp": str(restored_stamps[-1]) if stamps.size else None,
        "zero_cell_count": int(zero_count),
        "negative_cell_count": int(negative_count),
        "values_sha256": restored_hash.hexdigest(),
        "roundtrip_verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PROJECT.parent / "Datas_m_3039_H")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "prepared")
    parser.add_argument("--eval-end", type=date.fromisoformat, default=DEFAULT_EVAL_END)
    args = parser.parse_args()
    if args.eval_end < EVAL_START:
        parser.error("--eval-end는 2024-01-01 이후여야 합니다.")
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("원본과 별도의 출력 디렉터리를 지정해야 합니다.")
    if output.exists():
        parser.error(f"새 출력 디렉터리를 지정해 주세요. {output}")

    try:
        paths = index_csv_files(source)
        train, train_report = describe_period(paths, TRAIN_START, TRAIN_END)
        evaluation, eval_report = describe_period(paths, EVAL_START, args.eval_end)
        if not train:
            raise ValueError("학습 기간에 관측 파일이 없습니다.")
        print(f"원본 CSV {len(paths):,}개, 학습 {len(train):,}개, 평가 {len(evaluation):,}개", flush=True)
        report = {
            "source_directory": str(source),
            "source_csv_count": len(paths),
            "outside_requested_periods": len(paths) - len(train) - len(evaluation),
            "eval_end": args.eval_end.isoformat(),
            "time_axis": "filename interval start",
            "slots_per_day": len(HOURS),
            "grid_index_base": 0,
            "axis_order": ["time", "row", "column"],
            "train": train_report,
            "evaluation": eval_report,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temp:
            staging = Path(temp) / "prepared"
            staging.mkdir()
            train_report.update(save_split(staging, "train", train))
            eval_report.update(save_split(staging, "eval", evaluation))
            (staging / "preprocessing_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            staging.rename(output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"전처리 오류 (행·열은 0부터 시작). {error}\n")

    for label, split in (("학습", train_report), ("평가", eval_report)):
        print(
            f"{label} {tuple(split['shape'])} / {split['observed_start']} ~ "
            f"{split['observed_end']} / 누락 {split['missing_slot_count']:,} / "
            f"중복 {split['duplicate_timestamp_count']} / 저장 후 검증 완료"
        )
    if not eval_report["complete_coverage"]:
        print("요청 평가 기간 일부에 관측이 없습니다. 누락 시각은 preprocessing_report.json에 기록했습니다.")
    print(f"저장 위치 {output}")


if __name__ == "__main__":
    main()
