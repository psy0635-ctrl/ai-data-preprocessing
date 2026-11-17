"""Train and score a 43-input Dense model with known target-calendar features."""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

import evaluate_final as final
import validate_models as models

OUTPUT = models.PROJECT / "calendar_dense_experiment"
MODEL = "calendar_dense"
WIDTHS = (12, 7, 12)
WIDTH = models.LOOKBACK + sum(WIDTHS)
PARAMETERS = 961 + sum(WIDTHS) * models.HIDDEN_UNITS[0]
REFERENCE = {"2023": final.REFERENCE, "2024": final.OUTPUT}
PHASE_DIR = {"2023": "validation_2023", "2024": "test_2024"}
PREDICTION_FILES = ("prediction_timestamps.npy", f"{MODEL}_predictions.npy")
COMMAND = (
    "docker exec --workdir /workspace -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=2 "
    "dlproject python3 -B -u python/Floating_population/calendar_dense_experiment.py"
)


def encode_calendar(target_times):
    calendar = np.zeros((len(target_times), sum(WIDTHS)), dtype=np.float32)
    for index, stamp in enumerate(target_times.astype(object)):
        calendar[index, stamp.hour // 2] = 1
        calendar[index, WIDTHS[0] + stamp.weekday()] = 1
        calendar[index, sum(WIDTHS[:2]) + stamp.month - 1] = 1
    return calendar


def feature_audit(fit_times, forecast_times, normalized, fit_calendar, forecast_calendar):
    train_targets = fit_times[models.LOOKBACK:]
    if not np.array_equal(train_targets, fit_times[models.LOOKBACK - 1:-1] + models.STEP):
        raise ValueError("학습 창의 마지막 시각과 예측 대상 시각이 2시간 간격이 아닙니다.")
    for calendar in (fit_calendar, forecast_calendar):
        for group in np.split(calendar, np.cumsum(WIDTHS)[:-1], axis=1):
            if not np.all((group == 0) | (group == 1)) or not np.all(group.sum(axis=1) == 1):
                raise ValueError("달력 원-핫 그룹에 하나의 1이 있어야 합니다.")

    windows = np.lib.stride_tricks.sliding_window_view(normalized, models.LOOKBACK + 1, axis=1)
    first_targets = np.random.default_rng(models.SEED).permutation(len(train_targets))[:models.TARGETS_PER_BATCH]
    first_batch = models.make_dense_batch(windows, first_targets, fit_calendar)
    if first_batch.shape != (models.LOCATION_COUNT * models.TARGETS_PER_BATCH, WIDTH + 1):
        raise ValueError("실제 학습 배치의 입력 폭이 예상과 다릅니다.")

    def describe(target, previous, features, expected_indices):
        np.testing.assert_array_equal(np.flatnonzero(features[models.LOOKBACK:]), expected_indices)
        if target - previous != models.STEP:
            raise ValueError("날짜 경계 사례가 22:00 → 다음 00:00이 아닙니다.")
        return {
            "last_input_timestamp": str(previous),
            "target_timestamp": str(target),
            "one_hot_active_indices_0based_in_31_calendar_inputs": list(expected_indices),
            "one_hot_active_indices_0based_in_43_total_inputs": [
                models.LOOKBACK + index for index in expected_indices
            ],
            "example_input_43": features.tolist(),
        }

    # 2021-07-01 00:00 was Thursday. The preceding window ends in June.
    july = np.datetime64("2021-07-01T00:00")
    june = np.datetime64("2021-06-30T22:00")
    selected = int(train_targets.searchsorted(july))
    if train_targets[selected] != july or fit_times[selected + models.LOOKBACK - 1] != june:
        raise ValueError("학습 자료에서 월 경계 점검 시각을 찾지 못했습니다.")
    location = 22 * models.GRID_SHAPE[1] + 20
    boundary_batch = models.make_dense_batch(windows, [selected], fit_calendar)
    np.testing.assert_array_equal(boundary_batch[location, :models.LOOKBACK],
                                  normalized[location, selected:selected + models.LOOKBACK])
    shift = models.LOOKBACK
    np.testing.assert_array_equal(boundary_batch[location, WIDTH], normalized[location, selected + shift])
    inference_input = np.empty((models.LOCATION_COUNT, WIDTH), dtype=np.float32)
    inference_input[:, :shift] = normalized[:, -shift:]
    inference_input[:, shift:] = forecast_calendar[0]
    first_forecast = forecast_times[0]
    if first_forecast == np.datetime64("2023-01-01T00:00"):
        first_active = (0, 18, 19)  # Sunday, January
    elif first_forecast == np.datetime64("2024-01-01T00:00"):
        first_active = (0, 12, 19)  # Monday, January
    else:
        raise ValueError("연도 경계의 첫 예측 시각이 예상과 다릅니다.")
    return {
        "location_row_col_0based": [22, 20],
        "target_calendar_source": "target timestamp in fit and forecast",
        "training_input_shape": list(first_batch[:, :WIDTH].shape),
        "training_label_shape": list(first_batch[:, WIDTH].shape),
        "first_prediction_input_shape": list(inference_input.shape),
        "fit_calendar_shape": list(fit_calendar.shape),
        "forecast_calendar_shape": list(forecast_calendar.shape),
        "training_month_boundary": describe(july, june, boundary_batch[location, :WIDTH], (0, 15, 25)),
        "first_forecast_year_boundary": describe(first_forecast, fit_times[-1], inference_input[location], first_active),
    }


def expected_times(year):
    return np.arange(models.ORIGIN + models.STEP, models.VALID_END + models.STEP, models.STEP) if year == "2023" else final.EXPECTED_TIMES


def existing_score_paths(year):
    if year == "2023":
        return REFERENCE[year], "validation_metrics.json"
    return REFERENCE[year] / "evaluation", "evaluation_metrics.json"


def stage_settings(year):
    original = final.read_json(REFERENCE[year] / "run_config.json")["dense"]
    retained = {key: value for key, value in original.items() if key not in (
        "parameter_count", "settings_rationale", "zero_std_positions_with_scale_one"
    )}
    changes = {
        "input_width": {"old": models.LOOKBACK, "new": WIDTH},
        "calendar_one_hot_widths": {"hour_of_day": 12, "weekday": 7, "month": 12},
        "calendar_target_alignment": "target timestamp",
        "parameter_count": {"old": original["parameter_count"], "new": PARAMETERS},
    }
    if year == "2024":
        prior = OUTPUT / PHASE_DIR["2023"]
        if not (prior / "evaluation" / "metrics.json").is_file():
            raise ValueError("2023년 실험을 채점한 뒤 같은 설정으로 2024년 예측을 생성하세요.")
        recorded = final.read_json(prior / "run_config.json")
        if retained != recorded["retained_dense_settings"] or changes != recorded["changes_from_original_dense"]:
            raise ValueError("2023년 기록과 2024년 실행 설정이 일치하지 않습니다.")
    return retained, changes


def forecast(year, output):
    if output.exists():
        raise ValueError(f"예측 결과 경로가 이미 있습니다. {output}")
    retained, changes = stage_settings(year)
    if year == "2023":
        values, times, cutoff, _, targets = models.load_training_split(final.PREPARED)
    else:
        values, times = models.load_prepared_training(final.PREPARED)
        cutoff, targets = len(times), expected_times(year)
    fit_values, fit_times = values[:cutoff], times[:cutoff]
    if not np.isfinite(fit_values).all() or fit_times[-1] + models.STEP != targets[0]:
        raise ValueError("적합 자료의 값 또는 예측 시작 시각을 확인하세요.")
    normalized, mean, scale, constants = models.normalize_for_fit(fit_values.reshape(cutoff, models.LOCATION_COUNT))
    fit_calendar = encode_calendar(fit_times[models.LOOKBACK:])
    forecast_calendar = encode_calendar(targets)
    audit = feature_audit(fit_times, targets, normalized, fit_calendar, forecast_calendar)
    print(f"{year} 학습 {cutoff:,}구간, 예측 {len(targets):,}구간. 입력 확인 {audit['training_input_shape']} "
          f"/ 예측 입력 {audit['first_prediction_input_shape']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".calendar-predict-", dir=output.parent) as tmp:
        stage = Path(tmp) / output.name
        stage.mkdir()
        model, history = models.fit_dense(normalized, fit_calendar)
        if model.count_params() != PARAMETERS or model.input_shape[-1] != WIDTH:
            raise ValueError("달력 Dense 매개변수 수 또는 입력 폭이 예상과 다릅니다.")
        model.save(stage / "calendar_dense.keras")
        np.savez(stage / "dense_scaling.npz", mean=mean.reshape(models.GRID_SHAPE), scale=scale.reshape(models.GRID_SHAPE))
        models.write_csv(stage / "fit_history.csv", ("epoch", "sample_count", "normalized_mse"), history)
        predictions = models.predict_dense(model, normalized, mean, scale, len(targets), forecast_calendar)
        np.save(stage / f"{MODEL}_predictions.npy", predictions, allow_pickle=False)
        np.save(stage / "prediction_timestamps.npy", targets, allow_pickle=False)
        del predictions
        _, readback, checks = models.verify_saved_predictions(stage, targets, (MODEL,))
        del readback  # Release the mapped prediction file before renaming on the Windows bind mount.
        models.save_json(stage / "input_verification.json", audit)
        models.save_json(stage / "run_config.json", {
            "model": MODEL,
            "experiment": "follow-up for official multi-information input guideline",
            "earlier_2024_experiment_scores_already_known": True,
            "feature_specification": "fixed from official input requirements and retained 2021–2023 settings",
            "retained_dense_settings": retained,
            "changes_from_original_dense": changes,
            "zero_std_scale_one_count": constants,
            "fit_start": str(fit_times[0]),
            "fit_end": str(fit_times[-1]),
            "fit_slot_count": len(fit_times),
            "forecast_origin": str(fit_times[-1]),
            "start": str(targets[0]),
            "end": str(targets[-1]),
            "slot_count": len(targets),
            "grid_index_base": 0,
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "command": f"{COMMAND} {year} forecast",
        })
        models.save_json(stage / "forecast_verification.json", {
            "prediction_file_sha256": {name: final.digest(stage / name) for name in PREDICTION_FILES},
            "predictions": checks,
            "saved_and_verified_utc": final.utc_now(),
            "expected_timestamps_exact_match": True,
        })
        stage.rename(output)
    print(f"{year} 예측 저장 및 재검증 완료. {output}", flush=True)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def score(year, output):
    result = output / "evaluation"
    if result.exists():
        raise ValueError(f"채점 결과가 이미 있습니다. {result}")
    manifest = final.read_json(output / "forecast_verification.json")
    stamps, predictions, checks = models.verify_saved_predictions(output, expected_times(year), (MODEL,))
    if checks != manifest["predictions"] or any(
        final.digest(output / name) != manifest["prediction_file_sha256"][name] for name in PREDICTION_FILES
    ):
        raise ValueError("저장 예측과 파일 검증 기록이 일치하지 않습니다.")
    if not np.array_equal(stamps, np.load(REFERENCE[year] / "prediction_timestamps.npy", allow_pickle=False)):
        raise ValueError("기존 결과와 새 결과의 예측 시각이 일치하지 않습니다.")
    verification = {
        "forecast_verified_and_saved_utc": manifest["saved_and_verified_utc"],
        "forecast_rechecked_before_opening_actual_utc": final.utc_now(),
        "prediction_file_sha256": manifest["prediction_file_sha256"],
        "predictions": checks,
        "grid_index_base": 0,
        "command": f"{COMMAND} {year} score",
    }
    print(f"{year} 저장 예측과 기존 평가 시각 일치. 이제 채점 실제값을 엽니다.", flush=True)
    if year == "2023":
        values, times, start, end, _ = models.load_training_split(final.PREPARED)
        if not np.array_equal(stamps, times[start:end]):
            raise ValueError("2023년 실제값 시각이 일치하지 않습니다.")
        actual = values[start:end]
        verification["actual_timestamps"] = {
            "exact_match": True, "missing_timestamp_count": 0, "duplicate_timestamp_count": 0,
            "positional_mismatch_count": 0,
        }
        verification["actual_values"] = {
            "shape": list(actual.shape), "nan_count": int(np.count_nonzero(np.isnan(actual))),
            "infinity_count": int(np.count_nonzero(np.isinf(actual))),
        }
    else:
        actual = final.read_scoring_data(verification)
    if actual.shape != (len(stamps), *models.GRID_SHAPE):
        raise ValueError("평가 실제값 배열의 크기가 일치하지 않습니다.")
    verification["actual_checked_utc"] = final.utc_now()

    existing, filename = existing_score_paths(year)
    earlier = final.read_json(existing / filename)
    if earlier["start"] != str(stamps[0]) or earlier["end"] != str(stamps[-1]) or earlier["slot_count"] != len(stamps):
        raise ValueError("기존 점수의 기간이 동일하지 않습니다.")
    with TemporaryDirectory(prefix=".calendar-score-", dir=output) as tmp:
        stage = Path(tmp) / "evaluation"
        stage.mkdir()
        new, _, counts = models.score_predictions(stage, predictions, actual, stamps)
        if counts != earlier["actual_cell_counts"] or any(
            item["cell_count"] != actual.size for item in earlier["overall"].values()
        ):
            raise ValueError("기존 점수와 새 점수의 표본 수가 일치하지 않습니다.")
        all_scores = {**earlier["overall"], **new}
        neg = checks[MODEL]["negative_prediction_count"]
        checks[MODEL]["negative_prediction_fraction"] = neg / actual.size
        models.save_json(stage / "metrics.json", {
            "start": str(stamps[0]), "end": str(stamps[-1]), "slot_count": len(stamps),
            "cell_count": actual.size, "actual_cell_counts": counts, "new_predictions": checks[MODEL],
            "overall": all_scores, "existing_metrics_source": str(existing / filename),
        })
        models.save_json(stage / "verification.json", verification)
        models.write_csv(stage / "overall_comparison.csv", ("model", "cell_count", "mae", "rmse"),
                         ((name, m["cell_count"], m["mae"], m["rmse"]) for name, m in all_scores.items()))
        for table in ("monthly_mae.csv", "mae_by_actual.csv"):
            fields, new_rows = read_csv(stage / table)
            earlier_fields, old_rows = read_csv(existing / table)
            if fields != earlier_fields:
                raise ValueError(f"기존 점수표 형식이 일치하지 않습니다. {table}")
            models.write_csv(stage / table, fields, ([row[k] for k in fields] for row in old_rows + new_rows))
        stage.rename(result)
    print(f"{year} 달력 Dense MAE {new[MODEL]['mae']:.6f}, RMSE {new[MODEL]['rmse']:.6f}, "
          f"음수 {neg:,}개 ({neg / actual.size:.4%}). {result}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("year", choices=tuple(PHASE_DIR))
    parser.add_argument("action", choices=("forecast", "score"))
    args = parser.parse_args()
    output = OUTPUT / PHASE_DIR[args.year]
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    try:
        if args.action == "forecast":
            forecast(args.year, output)
        else:
            score(args.year, output)
    except (OSError, ValueError, AssertionError) as error:
        parser.exit(1, f"달력 Dense 실험 오류. {error}\n")


if __name__ == "__main__":
    main()
