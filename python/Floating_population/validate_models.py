"""Fit both models through 2022 and run fixed-origin validation in 2023."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import tensorflow as tf

from explore_train import plt, use_korean_font, write_csv

PROJECT = Path(__file__).resolve().parent
GRID_SHAPE = (72, 49)
LOCATION_COUNT = int(np.prod(GRID_SHAPE))
LOOKBACK = 12
STEP = np.timedelta64(2, "h")
ORIGIN = np.datetime64("2022-12-31T22:00")
VALID_END = np.datetime64("2023-05-31T22:00")
FIRST_TRAIN = np.datetime64("2021-06-01T00:00")
LAST_TRAIN = np.datetime64("2023-12-31T22:00")
EPOCHS = 3
TARGETS_PER_BATCH = 4
SEED = 42
LEARNING_RATE = 0.001
HIDDEN_UNITS = (32, 16)
LABELS = {"weekday_slot_mean": "요일·시간대 평균", "dense": "Dense"}
COLORS = {"weekday_slot_mean": "#2475a8", "dense": "#c45729"}


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_prepared_training(prepared):
    values = np.load(prepared / "train_values.npy", mmap_mode="r", allow_pickle=False)
    timestamps = np.load(prepared / "train_timestamps.npy", allow_pickle=False)
    expected = np.arange(FIRST_TRAIN, LAST_TRAIN + STEP, STEP)
    if (
        values.shape != (11328, *GRID_SHAPE)
        or values.dtype != np.dtype("float64")
        or timestamps.shape != (len(values),)
        or timestamps.dtype != np.dtype("datetime64[m]")
        or not np.array_equal(timestamps, expected)
    ):
        raise ValueError("전처리 학습 배열의 크기, 시각 범위, 정렬 또는 2시간 간격이 예상과 다릅니다.")
    return values, timestamps


def load_training_split(prepared):
    values, timestamps = load_prepared_training(prepared)
    forecast_times = np.arange(ORIGIN + STEP, VALID_END + STEP, STEP)
    fit_stop = int(timestamps.searchsorted(forecast_times[0]))
    validation_stop = fit_stop + len(forecast_times)
    if (
        fit_stop != 6948
        or timestamps[fit_stop - 1] != ORIGIN
        or len(forecast_times) != 1812
        or not np.array_equal(timestamps[fit_stop:validation_stop], forecast_times)
    ):
        raise ValueError("모델 적합 구간 또는 검증 시각을 확인해 주세요.")
    return values, timestamps, fit_stop, validation_stop, forecast_times


def weekday_slot(timestamps):
    return np.array([stamp.weekday() * 12 + stamp.hour // 2 for stamp in timestamps.astype(object)])


def fit_weekday_means(fit_values, fit_times):
    buckets = weekday_slot(fit_times)
    counts = np.bincount(buckets, minlength=84)
    if np.any(counts == 0):
        raise ValueError("모델 적합 기간에 비어 있는 요일·시간대 조합이 있습니다.")
    means = np.empty((84, *GRID_SHAPE), dtype=np.float64)
    for bucket in range(len(means)):
        means[bucket] = fit_values[buckets == bucket].mean(axis=0, dtype=np.float64)
    return means, counts


def normalize_for_fit(fit):
    mean = fit.mean(axis=0, dtype=np.float64)
    scale = fit.std(axis=0, dtype=np.float64)
    constant_mask = scale == 0
    scale[constant_mask] = 1.0
    normalized = np.empty((fit.shape[1], fit.shape[0]), dtype=np.float32)
    for start in range(0, len(fit), 256):
        stop = min(start + 256, len(fit))
        normalized[:, start:stop] = ((fit[start:stop] - mean) / scale).T
    return normalized, mean, scale, int(constant_mask.sum())


def make_dense_batch(windows, sampled_targets, target_features):
    histories = windows[:, sampled_targets, :]
    if target_features is None:
        return histories.reshape(-1, LOOKBACK + 1)
    input_width = LOOKBACK + target_features.shape[1]
    batch = np.empty((*histories.shape[:2], input_width + 1), dtype=np.float32)
    batch[..., :LOOKBACK] = histories[..., :LOOKBACK]
    batch[..., LOOKBACK:input_width] = target_features[sampled_targets]
    batch[..., input_width] = histories[..., LOOKBACK]
    return batch.reshape(-1, input_width + 1)


def fit_dense(normalized_fit, target_features=None):
    target_count = normalized_fit.shape[1] - LOOKBACK
    if target_features is not None and (target_features.ndim != 2 or len(target_features) != target_count):
        raise ValueError("Dense 학습 특성과 예측 대상 시각의 수가 일치하지 않습니다.")
    input_width = LOOKBACK if target_features is None else LOOKBACK + target_features.shape[1]
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(input_width,)),
            tf.keras.layers.Dense(HIDDEN_UNITS[0], activation="tanh"),
            tf.keras.layers.Dense(HIDDEN_UNITS[1], activation="tanh"),
            tf.keras.layers.Dense(1, activation="linear"),
        ],
        name="shared_position_dense",
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    optimizer.build(model.trainable_variables)

    @tf.function(input_signature=[tf.TensorSpec((None, input_width + 1), tf.float32)])
    def train_batch(samples):
        with tf.GradientTape() as tape:
            prediction = model(samples[:, :input_width], training=True)[:, 0]
            loss = tf.reduce_mean(tf.square(prediction - samples[:, input_width]))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    rng = np.random.default_rng(SEED)
    # This view covers all fit-only windows. Only the current batch is copied.
    windows = np.lib.stride_tricks.sliding_window_view(normalized_fit, LOOKBACK + 1, axis=1)
    history = []
    for epoch in range(1, EPOCHS + 1):
        started = perf_counter()
        targets = rng.permutation(windows.shape[1])
        loss_sum, seen = 0.0, 0
        for start in range(0, len(targets), TARGETS_PER_BATCH):
            sampled_targets = targets[start : start + TARGETS_PER_BATCH]
            batch = make_dense_batch(windows, sampled_targets, target_features)
            loss_sum += float(train_batch(batch).numpy()) * len(batch)
            seen += len(batch)
        mse = loss_sum / seen
        if not np.isfinite(mse) or seen != windows.shape[0] * windows.shape[1]:
            raise ValueError("Dense 학습 배치의 손실 또는 표본 수를 확인해 주세요.")
        history.append((epoch, seen, mse))
        print(f"Dense epoch {epoch}/{EPOCHS} | {seen:,} samples | MSE {mse:.6f} | {perf_counter() - started:.1f}s",
              flush=True)
    return model, history


def predict_dense(model, normalized_fit, mean, scale, count, target_features=None):
    if target_features is not None and (target_features.ndim != 2 or len(target_features) != count):
        raise ValueError("Dense 예측 특성과 예측 대상 시각의 수가 일치하지 않습니다.")
    input_width = LOOKBACK if target_features is None else LOOKBACK + target_features.shape[1]

    @tf.function(input_signature=[tf.TensorSpec((None, input_width), tf.float32)])
    def infer(inputs):
        return model(inputs, training=False)[:, 0]

    window = normalized_fit[:, -LOOKBACK:].copy()
    inputs = window if target_features is None else np.empty((len(mean), input_width), dtype=np.float32)
    flat_prediction = np.empty((count, len(mean)), dtype=np.float64)
    for index in range(count):
        if target_features is not None:
            inputs[:, :LOOKBACK] = window
            inputs[:, LOOKBACK:] = target_features[index]
        next_normalized = infer(inputs).numpy()
        flat_prediction[index] = next_normalized * scale + mean
        window[:, :-1] = window[:, 1:]
        window[:, -1] = next_normalized
    return flat_prediction.reshape(count, *GRID_SHAPE)


def verify_saved_predictions(output, observed_times, model_names=LABELS):
    times = np.load(output / "prediction_timestamps.npy", allow_pickle=False)
    if (
        times.shape != observed_times.shape
        or times.dtype != np.dtype("datetime64[m]")
        or not np.array_equal(times, observed_times)
        or not np.all(np.diff(times) == STEP)
    ):
        raise ValueError("저장한 예측 시각이 검증 시각과 일치하지 않습니다.")
    predictions, checks = {}, {}
    for name in model_names:
        array = np.load(output / f"{name}_predictions.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != (len(times), *GRID_SHAPE) or array.dtype != np.dtype("float64"):
            raise ValueError(f"예측 배열 크기 또는 자료형 오류. {name} {array.shape} {array.dtype}")
        nan_count = int(np.count_nonzero(np.isnan(array)))
        infinity_count = int(np.count_nonzero(np.isinf(array)))
        if nan_count or infinity_count:
            raise ValueError(f"예측 배열에 NaN 또는 무한값이 있습니다. {name} {nan_count=} {infinity_count=}")
        predictions[name] = array
        checks[name] = {
            "shape": list(array.shape),
            "nan_count": nan_count,
            "infinity_count": infinity_count,
            "negative_prediction_count": int(np.count_nonzero(array < 0)),
        }
    return times, predictions, checks


def score_predictions(output, predictions, actual, times):
    if not np.isfinite(actual).all():
        raise ValueError("실제값에 NaN 또는 무한값이 있습니다.")
    groups = {"zero": actual == 0, "positive": actual > 0}
    counts = {name: int(mask.sum()) for name, mask in groups.items()}
    month_index = times.astype("datetime64[M]")
    months = np.unique(month_index)
    overall, monthly_rows, subgroup_rows = {}, [], []
    for name, predicted in predictions.items():
        error = predicted - actual
        abs_error = np.abs(error)
        overall[name] = {
            "mae": float(abs_error.mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "cell_count": actual.size,
        }
        for month in months:
            month_error = abs_error[month_index == month]
            monthly_rows.append((name, str(month), len(month_error), month_error.size, float(month_error.mean())))
        for group, mask in groups.items():
            count = counts[group]
            group_mae = float(abs_error[mask].mean()) if count else None
            subgroup_rows.append((name, group, count, group_mae))
    write_csv(output / "monthly_mae.csv", ("model", "month", "slot_count", "cell_count", "mae"), monthly_rows)
    write_csv(output / "mae_by_actual.csv", ("model", "actual_group", "cell_count", "mae"), subgroup_rows)
    return overall, monthly_rows, counts


def plot_monthly(path, monthly_rows, *, title="2023년 고정 기준 시각 검증 · 월별 MAE"):
    fig, ax = plt.subplots(figsize=(8.2, 4.2), layout="constrained")
    for name, label in LABELS.items():
        rows = [row for row in monthly_rows if row[0] == name]
        months = [f"{int(row[1][-2:])}월" for row in rows]
        scores = [row[-1] for row in rows]
        ax.plot(months, scores, "o-", linewidth=2, color=COLORS[name], label=label)
        for month, score in zip(months, scores):
            ax.annotate(f"{score:.2f}", (month, score), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9)
    ax.set_title(title, loc="left")
    ax.set_ylabel("MAE (추정 인구 원본 값)")
    ax.set_ylim(0, 1.13 * max(row[-1] for row in monthly_rows))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_representatives(path, representatives_path, times, actual, predictions, *, label="2023년 내부 검증"):
    from matplotlib.dates import DateFormatter, DayLocator, MonthLocator

    with representatives_path.open(encoding="utf-8-sig", newline="") as stream:
        positions = [(int(row["row_0based"]), int(row["col_0based"])) for row in csv.DictReader(stream)]
    if len(positions) != 3 or len(set(positions)) != 3 or any(
        not (0 <= row < GRID_SHAPE[0] and 0 <= col < GRID_SHAPE[1]) for row, col in positions
    ):
        raise ValueError("탐색 단계의 대표 위치 3개를 확인해 주세요.")

    end_week = times >= (times[-1].astype("datetime64[D]") - np.timedelta64(6, "D"))
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 8.0), sharey="row", layout="constrained")
    fig.suptitle(f"{label} · 대표 위치의 실제값과 예측값", fontsize=16)
    for index, (row, col) in enumerate(positions):
        series = {
            "actual": actual[:, row, col],
            **{name: array[:, row, col] for name, array in predictions.items()},
        }
        for col_index, mask in enumerate((slice(None), end_week)):
            axis = axes[index, col_index]
            for name in (*LABELS, "actual"):
                color = COLORS[name] if name in LABELS else "#18212f"
                label = LABELS.get(name, "실제값")
                axis.plot(times[mask], series[name][mask], color=color, label=label,
                          linewidth=0.95 if name == "actual" else 1.15,
                          alpha=0.86, linestyle="--" if name == "dense" else "-")
            axis.grid(axis="y", alpha=0.22)
            axis.set_xlim(times[mask][0], times[-1])
            axis.tick_params(axis="x", labelbottom=index == len(positions) - 1)
            if col_index == 0:
                axis.xaxis.set_major_locator(MonthLocator())
                axis.xaxis.set_major_formatter(DateFormatter("%m월"))
                axis.set_ylabel("추정 인구 원본 값")
                axis.set_title(f"행렬 위치 ({row}, {col}) · 기간 전체", loc="left")
            else:
                axis.xaxis.set_major_locator(DayLocator())
                axis.xaxis.set_major_formatter(DateFormatter("%m-%d"))
                axis.set_title(f"행렬 위치 ({row}, {col}) · 마지막 7일", loc="left")
    axes[0, 1].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.11), ncols=3)
    year = np.datetime_as_string(times[0], unit="Y")
    fig.supxlabel(f"{year}년 날짜 · 행과 열의 인덱스는 0부터 시작")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def fit_and_save_predictions(output, fit_values, fit_times, forecast_times):
    if not np.isfinite(fit_values).all():
        raise ValueError("모델 적합 범위에 NaN 또는 무한값이 있습니다.")
    baseline_means, bucket_counts = fit_weekday_means(fit_values, fit_times)
    np.save(output / "weekday_slot_means.npy", baseline_means.reshape(7, 12, *GRID_SHAPE), allow_pickle=False)
    baseline = baseline_means[weekday_slot(forecast_times)]
    np.save(output / "weekday_slot_mean_predictions.npy", baseline, allow_pickle=False)
    del baseline

    prepared_fit = fit_values.reshape(len(fit_values), LOCATION_COUNT)
    normalized, mean, scale, constant_count = normalize_for_fit(prepared_fit)
    model, fit_history = fit_dense(normalized)
    model.save(output / "dense_model.keras")
    np.savez(output / "dense_scaling.npz", mean=mean.reshape(GRID_SHAPE), scale=scale.reshape(GRID_SHAPE))
    write_csv(output / "dense_fit_history.csv", ("epoch", "sample_count", "normalized_mse"), fit_history)
    dense = predict_dense(model, normalized, mean, scale, len(forecast_times))
    np.save(output / "dense_predictions.npy", dense, allow_pickle=False)
    np.save(output / "prediction_timestamps.npy", forecast_times, allow_pickle=False)
    return model.count_params(), bucket_counts, constant_count


def run_validation(prepared, representatives, output):
    values, times, fit_stop, valid_stop, forecast_times = load_training_split(prepared)
    print(f"모델 적합 {fit_stop:,}개 구간, 내부 검증 {len(forecast_times):,}개 예정", flush=True)
    parameter_count, bucket_counts, constant_count = fit_and_save_predictions(
        output, values[:fit_stop], times[:fit_stop], forecast_times
    )
    validation_times, predictions, checks = verify_saved_predictions(output, times[fit_stop:valid_stop])
    print("두 모델의 저장 예측 배열 및 시각 검증 완료. 2023년 실제값으로 채점합니다.", flush=True)
    # Slice validation values only after both forecasts are persisted and verified.
    actual = values[fit_stop:valid_stop]
    overall, monthly_rows, group_counts = score_predictions(output, predictions, actual, validation_times)
    save_json(
        output / "validation_metrics.json",
        {
            "start": str(validation_times[0]),
            "end": str(validation_times[-1]),
            "slot_count": len(validation_times),
            "missing_timestamp_count": 0,
            "duplicate_timestamp_count": 0,
            "actual_cell_counts": group_counts,
            "predictions": checks,
            "overall": overall,
        },
    )
    save_json(
        output / "run_config.json",
        {
            "input_files": [str(prepared / name) for name in ("train_values.npy", "train_timestamps.npy")],
            "input_shape": list(values.shape),
            "fit_start": str(times[0]),
            "fit_end": str(times[fit_stop - 1]),
            "fit_slot_count": fit_stop,
            "forecast_origin": str(ORIGIN),
            "validation_start": str(validation_times[0]),
            "validation_end": str(validation_times[-1]),
            "validation_slot_count": len(validation_times),
            "grid_shape": list(GRID_SHAPE),
            "grid_index_base": 0,
            "baseline": {
                "method": "per-position arithmetic mean for each weekday and 2-hour start time",
                "weekday_order": "Monday to Sunday",
                "slot_order": "00:00 to 22:00",
                "fit_slot_counts": bucket_counts.reshape(7, 12).tolist(),
            },
            "dense": {
                "lookback_slots": LOOKBACK,
                "lookback_hours": LOOKBACK * 2,
                "one_step_horizon_hours": 2,
                "forecast_mode": "recursive fixed origin",
                "shared_across_positions": True,
                "hidden_units": list(HIDDEN_UNITS),
                "hidden_activation": "tanh",
                "output_activation": "linear",
                "parameter_count": parameter_count,
                "settings_rationale": {
                    "layers": "961 parameters for a shared nonlinear univariate model",
                    "activation": "bounded hidden activations keep recursive extrapolation finite",
                    "training": "three fixed epochs of Adam for initial comparison without validation tuning",
                    "batching": "four shuffled fit target instants across all locations per batch",
                },
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "loss": "MSE on standardized values",
                "normalization": "per-position fit-only mean and population standard deviation",
                "zero_std_positions_with_scale_one": constant_count,
                "output_clipping": False,
                "epochs": EPOCHS,
                "targets_per_batch": TARGETS_PER_BATCH,
                "max_batch_samples": TARGETS_PER_BATCH * LOCATION_COUNT,
                "seed": SEED,
                "early_stopping": False,
            },
            "numpy_version": np.__version__,
            "tensorflow_version": tf.__version__,
        },
    )
    plot_monthly(output / "monthly_mae.png", monthly_rows)
    plot_representatives(output / "representative_predictions.png", representatives, validation_times, actual, predictions)
    for name, metrics in overall.items():
        print(f"{name}: MAE {metrics['mae']:.6f}, RMSE {metrics['rmse']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=PROJECT / "prepared")
    parser.add_argument("--representatives", type=Path, default=PROJECT / "exploration" / "representative_positions.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "validation_2023")
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    prepared = args.prepared_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(prepared):
        parser.error(f"전처리 폴더 밖의 새 출력 디렉터리를 지정해 주세요. {output}")

    try:
        use_korean_font(args.font)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".validate-", dir=output.parent) as temp:
            staging = Path(temp) / "validation"
            staging.mkdir()
            run_validation(prepared, args.representatives, staging)
            staging.rename(output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"내부 검증 오류. {error}\n")
    print(f"검증 결과 저장 위치 {output}")


if __name__ == "__main__":
    main()
