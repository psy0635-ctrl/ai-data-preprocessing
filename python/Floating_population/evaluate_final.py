"""Generate fixed-origin 2024 predictions, then score them in a separate command."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

import validate_models as models

PROJECT = models.PROJECT
PREPARED = PROJECT / "prepared"
REFERENCE = PROJECT / "validation_2023"
OUTPUT = PROJECT / "final_evaluation_2024"
START = np.datetime64("2024-01-01T00:00")
END = np.datetime64("2024-05-31T22:00")
EXPECTED_TIMES = np.arange(START, END + models.STEP, models.STEP)
EXPECTED_SHAPE = (1824, *models.GRID_SHAPE)
PRIMARY = "weekday_slot_mean"
COMPARISON = "dense"
SELECTION = {
    "primary": PRIMARY,
    "comparison": COMPARISON,
    "criterion": "MAE from 2023 internal validation",
    "reference": str(REFERENCE / "validation_metrics.json"),
}
CHART_LOCATIONS = [(22, 20), (15, 25), (7, 22)]
EXAMPLES = PROJECT / "exploration" / "representative_positions.csv"
PREDICTION_FILES = ("prediction_timestamps.npy", *(f"{name}_predictions.npy" for name in models.LABELS))
DOCKER_COMMAND = (
    "docker exec --workdir /workspace -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=2 "
    "dlproject python3 -B -W error::UserWarning -u python/Floating_population/evaluate_final.py"
)
FORECAST_COMMAND = f"{DOCKER_COMMAND} forecast"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_prior_validation():
    config = read_json(REFERENCE / "run_config.json")
    metrics = read_json(REFERENCE / "validation_metrics.json")
    dense_settings = config["dense"]
    expected = {
        "lookback_slots": models.LOOKBACK,
        "lookback_hours": models.LOOKBACK * 2,
        "one_step_horizon_hours": 2,
        "forecast_mode": "recursive fixed origin",
        "shared_across_positions": True,
        "hidden_units": list(models.HIDDEN_UNITS),
        "hidden_activation": "tanh",
        "output_activation": "linear",
        "optimizer": "Adam",
        "learning_rate": models.LEARNING_RATE,
        "loss": "MSE on standardized values",
        "normalization": "per-position fit-only mean and population standard deviation",
        "epochs": models.EPOCHS,
        "targets_per_batch": models.TARGETS_PER_BATCH,
        "max_batch_samples": models.TARGETS_PER_BATCH * models.LOCATION_COUNT,
        "seed": models.SEED,
        "output_clipping": False,
        "early_stopping": False,
    }
    differences = [name for name, value in expected.items() if dense_settings.get(name) != value]
    if differences:
        raise ValueError(f"2023년 Dense 설정과 차이가 있습니다. {', '.join(differences)}")
    if metrics["overall"][PRIMARY]["mae"] >= metrics["overall"][COMPARISON]["mae"]:
        raise ValueError("고정한 주 모델과 2023년 MAE 기록을 확인해 주세요.")
    return config, metrics


def forecast():
    if OUTPUT.exists():
        raise ValueError(f"예측 결과 경로가 이미 있습니다. {OUTPUT}")
    prior, prior_metrics = read_prior_validation()
    fit_values, fit_times = models.load_prepared_training(PREPARED)
    if fit_times[-1] + models.STEP != EXPECTED_TIMES[0] or len(EXPECTED_TIMES) != EXPECTED_SHAPE[0]:
        raise ValueError("최종 예측 기준 시각과 예측 시각 범위가 맞지 않습니다.")
    split_report = read_json(PREPARED / "preprocessing_report.json")["train"]
    if (
        split_report["missing_slot_count"] != 1812
        or split_report["missing_timestamps"][0] != "2021-01-01T00:00"
        or split_report["missing_timestamps"][-1] != "2021-05-31T22:00"
    ):
        raise ValueError("2021년 1~5월 미제공 기록을 확인해 주세요.")

    print(f"주 모델 {PRIMARY}. 선택 근거는 2023년 내부 검증 MAE입니다.", flush=True)
    print(f"전체 학습 {len(fit_times):,}개로 재적합. 예측 {len(EXPECTED_TIMES):,}개 예정.", flush=True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    with TemporaryDirectory(prefix=".final-forecast-", dir=PROJECT) as tmp:
        stage = Path(tmp) / OUTPUT.name
        stage.mkdir()
        count, buckets, constant_count = models.fit_and_save_predictions(
            stage, fit_values, fit_times, EXPECTED_TIMES
        )
        dense_settings = dict(prior["dense"], zero_std_positions_with_scale_one=constant_count)
        if count != dense_settings["parameter_count"]:
            raise ValueError("Dense 매개변수 수가 2023년 설정과 다릅니다.")
        saved_times, predictions, quality = models.verify_saved_predictions(stage, EXPECTED_TIMES)
        del predictions
        config = {
            "model_selection": dict(SELECTION, prior_mae={
                name: prior_metrics["overall"][name]["mae"] for name in models.LABELS
            }),
            "fit_start": str(fit_times[0]),
            "fit_end": str(fit_times[-1]),
            "fit_slot_count": len(fit_times),
            "fit_files": [str(PREPARED / name) for name in ("train_values.npy", "train_timestamps.npy")],
            "requested_fit_start": split_report["requested_start"],
            "unprovided_fit_dates": ["2021-01-01", "2021-05-31"],
            "unprovided_fit_slot_count": split_report["missing_slot_count"],
            "forecast_origin": str(fit_times[-1]),
            "evaluation_start": str(saved_times[0]),
            "evaluation_end": str(saved_times[-1]),
            "slot_count": len(saved_times),
            "grid_index_base": 0,
            "grid_shape": list(models.GRID_SHAPE),
            "baseline": dict(prior["baseline"], fit_slot_counts=buckets.reshape(7, 12).tolist()),
            "dense": dense_settings,
            "numpy_version": np.__version__,
            "tensorflow_version": tf.__version__,
            "forecast_command": FORECAST_COMMAND,
        }
        models.save_json(stage / "run_config.json", config)
        manifest = {
            "verified_and_saved_utc": utc_now(),
            "expected_timestamp_match": True,
            "missing_timestamp_count": 0,
            "duplicate_timestamp_count": 0,
            "predictions": quality,
            "prediction_file_sha256": {name: digest(stage / name) for name in PREDICTION_FILES},
        }
        models.save_json(stage / "forecast_verification.json", manifest)
        stage.rename(OUTPUT)
    print(f"두 모델의 저장 예측과 SHA-256 확인 완료. {OUTPUT}")


def timestamp_check(actual_times):
    result = {"shape": list(actual_times.shape), "dtype": str(actual_times.dtype)}
    if actual_times.ndim != 1 or actual_times.dtype != EXPECTED_TIMES.dtype:
        return dict(result, exact_match=False)
    unique, counts = np.unique(actual_times, return_counts=True)
    missing = np.setdiff1d(EXPECTED_TIMES, unique)
    common_count = min(len(actual_times), len(EXPECTED_TIMES))
    return dict(
        result,
        exact_match=bool(np.array_equal(actual_times, EXPECTED_TIMES)),
        missing_timestamps=missing.astype(str).tolist(),
        unexpected_timestamps=np.setdiff1d(unique, EXPECTED_TIMES).astype(str).tolist(),
        duplicate_timestamps=unique[counts > 1].astype(str).tolist(),
        missing_timestamp_count=int(missing.size),
        duplicate_timestamp_count=int(np.sum(counts - 1)),
        positional_mismatch_count=(
            int(np.count_nonzero(actual_times[:common_count] != EXPECTED_TIMES[:common_count]))
            + abs(len(actual_times) - len(EXPECTED_TIMES))
        ),
    )


def read_scoring_data(verification):
    actual_times = np.load(PREPARED / "eval_timestamps.npy", allow_pickle=False)
    verification["actual_timestamps"] = timestamp_check(actual_times)
    if not verification["actual_timestamps"]["exact_match"]:
        raise ValueError("평가 실제값의 시각 배열이 저장된 예측과 일치하지 않습니다.")

    actual = np.load(PREPARED / "eval_values.npy", mmap_mode="r", allow_pickle=False)
    quality = {"shape": list(actual.shape), "dtype": str(actual.dtype)}
    verification["actual_values"] = quality
    if actual.shape != EXPECTED_SHAPE or actual.dtype != np.dtype("float64"):
        raise ValueError("평가 실제값 배열의 크기 또는 자료형이 다릅니다.")
    quality["nan_count"] = int(np.count_nonzero(np.isnan(actual)))
    quality["infinity_count"] = int(np.count_nonzero(np.isinf(actual)))
    if quality["nan_count"] or quality["infinity_count"]:
        raise ValueError("평가 실제값 배열에 NaN 또는 무한값이 있습니다.")
    return actual


def comparison_rows(prior, prior_metrics, config, overall):
    rows = []
    for phase, settings, results, start, end in (
        ("2023 내부 검증", prior, prior_metrics["overall"], prior["validation_start"], prior["validation_end"]),
        ("2024 최종 평가", config, overall, config["evaluation_start"], config["evaluation_end"]),
    ):
        for name in models.LABELS:
            scores = results[name]
            rows.append((phase, models.LABELS[name], settings["fit_start"], settings["fit_end"], start, end,
                         scores["mae"], scores["rmse"]))
    return rows


def write_submission_summary(path, rows, config, summary, score_command):
    dense_check = summary["predictions"][COMPARISON]
    lines = [
        "# 유동인구 시계열 기말 프로젝트 · 2024년 최종 평가", "",
        "## 1. 모델과 기간", "",
        "2023년 내부 검증 MAE를 기준으로 요일·시간대 평균 모델을 주 모델로 고정했다. Dense는 비교 모델이다.",
        "제공본의 학습 관측은 2021년 6월 1일부터이며, 2021년 1~5월 1,812개 구간은 원본 미제공이다.",
        "기준 모델 평균과 Dense의 위치별 평균·표준편차를 전체 학습 11,328개 구간에서 다시 계산했다.",
        "Dense는 직전 12개 관측(24시간)으로 다음 2시간 구간을 예측한다. 2023년 12월 31일 "
        "22:00을 기준으로 시작해 이후에는 먼저 생성한 예측값을 입력 창에 넣었다.",
        "구조는 12 → 32 → 16 → 1, 은닉층 tanh, 선형 출력, Adam 0.001, 정규화 척도의 MSE, "
        "3 epoch, 시드 42다. 학습 배치당 최대 표본 수는 14,112개다.", "",
        "## 2. 내부 검증과 최종 평가 비교", "",
        "| 평가 | 모델 | 적합 기간 | 평가 기간 | MAE | RMSE |",
        "|---|---|---|---|---:|---:|",
    ]
    for phase, name, fit_start, fit_end, start, end, mae, rmse in rows:
        lines.append(
            f"| {phase} | {name} | {fit_start}~{fit_end} | {start}~{end} | {mae:.6f} | {rmse:.6f} |"
        )
    lines += [
        "", "## 3. 검증 기록", "",
        f"실제값과 두 예측은 각각 {tuple(summary['actual_shape'])}이다. "
        f"같은 시각과 0부터 세는 행·열을 연결해 {summary['cell_count']:,}개 셀을 비교했다.",
        "누락, 중복, 시각 불일치, NaN·무한값은 각각 0건이다.",
        f"Dense 음수 예측 {dense_check['negative_prediction_count']:,}개"
        f"({dense_check['negative_prediction_fraction']:.4%})를 원본 척도 점수에 포함했다.",
        "두 예측을 먼저 저장하고 SHA-256과 배열·시각을 확인한 후, 별도 채점 명령에서 2024년 실제값을 열었다. "
        "모델 선택에는 2023년 내부 검증 점수를 사용했고, 재적합·정규화·예측에는 2023년 말까지의 학습 자료를 사용했다.",
        "대표 행렬 위치는 (22, 20), (15, 25), (7, 22)다.", "",
        "## 4. 실행 명령과 결과 경로", "", "```powershell",
        config["forecast_command"], score_command, "```", "",
        f"- 모델 설정과 예측 파일은 `{OUTPUT}`",
        f"- 점수·그래프·대조 기록은 `{OUTPUT / 'evaluation'}`",
        "- 월별 점수와 실제값 0·양수별 점수는 각각 `monthly_mae.csv`, `mae_by_actual.csv`",
        "- 두 평가를 비교한 표는 `score_comparison.csv`",
        "- 저장 예측과 실제값 대조 기록은 `verification.json`", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def score(font):
    output = OUTPUT / "evaluation"
    if output.exists():
        raise ValueError(f"이미 채점한 결과가 있습니다. {output}")
    config = read_json(OUTPUT / "run_config.json")
    if any(config["model_selection"].get(key) != value for key, value in SELECTION.items()):
        raise ValueError("2024년 실제값을 열기 전 주·비교 모델 고정 기록을 확인해 주세요.")
    prior, prior_metrics = read_prior_validation()
    manifest = read_json(OUTPUT / "forecast_verification.json")
    timestamps, predictions, checks = models.verify_saved_predictions(OUTPUT, EXPECTED_TIMES)
    if {name: digest(OUTPUT / name) for name in PREDICTION_FILES} != manifest["prediction_file_sha256"]:
        raise ValueError("저장 예측의 SHA-256이 예측 생성 단계의 기록과 일치하지 않습니다.")
    if checks != manifest["predictions"]:
        raise ValueError("저장 예측의 품질 검사 결과가 예측 생성 단계의 기록과 다릅니다.")
    verification = {
        "model_selection": config["model_selection"],
        "forecast_verified_and_saved_utc": manifest["verified_and_saved_utc"],
        "forecast_checksum_reconfirmed_utc": utc_now(),
        "prediction_file_sha256": manifest["prediction_file_sha256"],
        "predictions": checks,
        "used_eval_values_for": ["scoring", "actual-versus-predicted plots"],
        "grid_index_base": 0,
    }
    print("저장 예측의 크기·시각·SHA-256 재확인 완료. 평가 실제값을 엽니다.", flush=True)
    try:
        actual = read_scoring_data(verification)
    except (OSError, ValueError) as error:
        models.save_json(OUTPUT / "evaluation_failure.json", dict(verification, error=str(error)))
        raise
    verification["actual_checked_utc"] = utc_now()
    models.use_korean_font(font)
    with EXAMPLES.open(encoding="utf-8-sig") as stream:
        locations = [(int(row["row_0based"]), int(row["col_0based"])) for row in csv.DictReader(stream)]
    if locations != CHART_LOCATIONS:
        raise ValueError("대표 행렬 위치 목록이 탐색 단계의 세 예시와 일치하지 않습니다.")

    with TemporaryDirectory(prefix=".final-score-", dir=OUTPUT) as tmp:
        stage = Path(tmp) / "evaluation"
        stage.mkdir()
        overall, monthly, counts = models.score_predictions(stage, predictions, actual, timestamps)
        for quality in checks.values():
            quality["negative_prediction_fraction"] = quality["negative_prediction_count"] / actual.size
        summary = {
            "fit_start": config["fit_start"],
            "fit_end": config["fit_end"],
            "forecast_origin": config["forecast_origin"],
            "start": str(timestamps[0]),
            "end": str(timestamps[-1]),
            "slot_count": len(timestamps),
            "cell_count": actual.size,
            "actual_shape": list(actual.shape),
            "actual_cell_counts": counts,
            "predictions": checks,
            "overall": overall,
        }
        rows = comparison_rows(prior, prior_metrics, config, overall)
        score_command = f"{DOCKER_COMMAND} score" + (f" --font {font}" if font else "")
        models.save_json(stage / "evaluation_metrics.json", summary)
        models.save_json(stage / "verification.json", dict(verification, score_command=score_command))
        models.write_csv(
            stage / "score_comparison.csv",
            ("evaluation", "model", "fit_start", "fit_end", "eval_start", "eval_end", "mae", "rmse"),
            rows,
        )
        write_submission_summary(stage / "submission_summary.md", rows, config, summary, score_command)
        models.plot_monthly(stage / "monthly_mae.png", monthly, title="2024년 최종 평가 · 월별 MAE")
        models.plot_representatives(
            stage / "representative_predictions.png", EXAMPLES, timestamps, actual, predictions,
            label="2024년 최종 평가",
        )
        stage.rename(output)
    for name, metrics in overall.items():
        print(f"{name}: MAE {metrics['mae']:.6f}, RMSE {metrics['rmse']:.6f}")
    print(f"채점 결과 저장 위치 {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    phases.add_parser("forecast")
    scoring = phases.add_parser("score")
    scoring.add_argument("--font", type=Path)
    args = parser.parse_args()
    try:
        if args.phase == "forecast":
            forecast()
        else:
            score(args.font)
    except (OSError, ValueError) as error:
        parser.exit(1, f"최종 평가 오류. {error}\n")


if __name__ == "__main__":
    main()
