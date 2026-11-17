"""Generate position statistics and time-pattern plots using the training split."""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fonts
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ft2font import FT2Font

PROJECT = Path(__file__).resolve().parent
WEEKDAYS = ("월", "화", "수", "목", "금", "토", "일")
HOURS = tuple(range(0, 24, 2))
SEASONS = {
    "봄": (3, 4, 5),
    "여름": (6, 7, 8),
    "가을": (9, 10, 11),
    "겨울": (12, 1, 2),
}
COLORS = ("#4E876F", "#3579A8", "#C08047", "#7A74A5")
POSITION_FIELDS = ("row_0based", "col_0based")
VALUE_LABEL = "추정 인구 원본 값 (평균)"


def use_korean_font(font_path):
    glyphs = set("행렬위치추정인구원본값평균요일월화수목금토일시간대구간시작봄여름가을겨울관측수학습자료계절색범위")
    if font_path:
        candidates = [str(font_path)]
    else:
        candidates = fonts.findSystemFonts()
    for path in candidates:
        charmap = FT2Font(path).get_charmap()
        if all(ord(glyph) in charmap for glyph in glyphs):
            fonts.fontManager.addfont(path)
            family = fonts.FontProperties(fname=path).get_name()
            plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})
            return
    raise ValueError("한글을 지원하는 글꼴을 --font로 지정해 주세요.")


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def select_positions(means, zero_fraction):
    flat_means = means.ravel()
    flat_zero = zero_fraction.ravel()
    positive = np.flatnonzero(flat_means > 0)
    if positive.size < 2:
        raise ValueError("특성이 다른 대표 위치를 선정할 양수 평균 위치가 부족합니다.")
    peak = positive[np.argmax(flat_means[positive])]
    candidates = positive[positive != peak]
    median = np.median(flat_means[candidates])
    middle = candidates[np.argmin(np.abs(flat_means[candidates] - median))]
    selected = [(peak, "학습 평균 최댓값"), (middle, "나머지 양수 평균 위치의 중앙값에 가장 가까움")]
    candidates = candidates[candidates != middle]
    if candidates.size:
        sparse = candidates[np.argmin(np.abs(flat_zero[candidates] - 0.95))]
        selected.append((sparse, "남은 양수 평균 위치 중 0 기록 비율 95%에 가장 가까움"))
    return [(np.unravel_index(index, means.shape), reason) for index, reason in selected]


def mean_label_format(means):
    magnitude = np.nanmax(np.abs(means))
    if magnitude < 0.01:
        return ".6f"
    if magnitude < 1:
        return ".3f"
    return ",.2f"


def heatmap(path, position, weekday_means):
    fig, ax = plt.subplots(figsize=(11, 5.2), layout="constrained")
    image = ax.imshow(weekday_means, cmap="viridis", aspect="auto")
    ax.set_title(f"행렬 위치 ({position[0]}, {position[1]}) · 요일 × 2시간대 평균", loc="left", pad=15)
    ax.set_title("행·열은 0부터", loc="right", pad=15, fontsize=10, color="#475569")
    ax.set_xticks(range(len(HOURS)), [f"{hour:02d}:00" for hour in HOURS])
    ax.set_yticks(range(len(WEEKDAYS)), WEEKDAYS)
    ax.set_xlabel("2시간 구간 시작 시각")
    ax.set_ylabel("요일")
    ax.tick_params(length=0)
    number_format = mean_label_format(weekday_means)
    for (row, col), value in np.ndenumerate(weekday_means):
        if np.isfinite(value):
            color = "#152030" if image.norm(value) > 0.56 else "white"
            label = format(value, number_format) if value else "0"
            ax.text(col, row, label, ha="center", va="center", color=color, fontsize=7)
    fig.colorbar(image, ax=ax, label=VALUE_LABEL, pad=0.025)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def seasonal_chart(path, groups, counts):
    fig, axes = plt.subplots(1, len(groups), figsize=(12, 4.6), squeeze=False)
    for axis, (position, means) in zip(axes[0], groups):
        bars = axis.bar(range(len(SEASONS)), means, color=COLORS, width=0.65)
        axis.set_title(f"행렬 위치 ({position[0]}, {position[1]})", pad=12)
        axis.set_xticks(range(len(SEASONS)), [f"{season}\nn={count:,}" for season, count in zip(SEASONS, counts)])
        number_format = mean_label_format(means)
        labels = [format(value, number_format) if np.isfinite(value) else "" for value in means]
        axis.bar_label(bars, labels=labels, padding=4)
        axis.margins(y=0.20)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.23)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", length=0)
    axes[0, 0].set_ylabel(VALUE_LABEL)
    fig.suptitle("행렬 위치별 계절 평균과 관측 수", x=0.075, y=0.97, ha="left", fontsize=15)
    fig.text(
        0.075,
        0.06,
        "행·열은 0부터 셉니다. 학습 자료는 2021년 6월부터 시작하므로 계절별 관측 연도와 관측 수가 다릅니다.",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.81, wspace=0.33)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def explore(prepared, output):
    values = np.load(prepared / "train_values.npy", mmap_mode="r", allow_pickle=False)
    timestamps = np.load(prepared / "train_timestamps.npy", allow_pickle=False)
    if (
        values.ndim != 3
        or values.shape[1:] != (72, 49)
        or timestamps.shape != (len(values),)
        or timestamps.dtype != np.dtype("datetime64[m]")
        or not len(timestamps)
        or np.any(timestamps[1:] <= timestamps[:-1])
        or timestamps[-1] >= np.datetime64("2024-01-01")
    ):
        raise ValueError("학습 격자와 시각을 확인해 주세요.")
    means = values.mean(axis=0, dtype=np.float64)
    zero_counts = np.count_nonzero(values == 0, axis=0)
    zero_fraction = zero_counts / len(values)
    write_csv(
        output / "position_stats.csv",
        (*POSITION_FIELDS, "observation_count", "mean", "zero_count", "zero_fraction"),
        ((*pos, len(values), mean, zero_counts[pos], zero_fraction[pos]) for pos, mean in np.ndenumerate(means)),
    )
    selected = select_positions(means, zero_fraction)
    write_csv(
        output / "representative_positions.csv",
        (*POSITION_FIELDS, "mean", "zero_fraction", "selection_reason"),
        ((*pos, means[pos], zero_fraction[pos], reason) for pos, reason in selected),
    )

    datetimes = timestamps.astype(object)
    weekdays = np.array([stamp.weekday() for stamp in datetimes])
    hours = np.array([stamp.hour for stamp in datetimes])
    months = np.array([stamp.month for stamp in datetimes])
    year_months = timestamps.astype("datetime64[M]")
    weekday_groups = {
        (day, hour): (weekdays == day) & (hours == hour)
        for day in range(len(WEEKDAYS))
        for hour in HOURS
    }
    season_groups = {season: np.isin(months, names) for season, names in SEASONS.items()}
    season_counts = [int(mask.sum()) for mask in season_groups.values()]

    weekday_rows, season_rows, season_chart_groups = [], [], []
    for (row, col), _ in selected:
        series = values[:, row, col]
        grid = np.full((len(WEEKDAYS), len(HOURS)), np.nan)
        for (day, hour), mask in weekday_groups.items():
            group = series[mask]
            mean = group.mean() if group.size else np.nan
            grid[day, hour // 2] = mean
            weekday_rows.append((row, col, WEEKDAYS[day], f"{hour:02d}:00", group.size, mean if group.size else ""))
        heatmap(output / f"weekday_hour_r{row:02d}_c{col:02d}.png", (row, col), grid)

        seasonal_means = []
        for season, mask in season_groups.items():
            group = series[mask]
            mean = group.mean() if group.size else np.nan
            seasonal_means.append(mean)
            month_labels = " ".join(np.unique(year_months[mask]).astype(str))
            season_rows.append((row, col, season, group.size, mean if group.size else "", month_labels))
        season_chart_groups.append(((row, col), seasonal_means))

    write_csv(
        output / "weekday_hour_summary.csv",
        (*POSITION_FIELDS, "weekday", "slot_start", "observation_count", "mean"),
        weekday_rows,
    )
    write_csv(
        output / "seasonal_summary.csv",
        (*POSITION_FIELDS, "season", "observation_count", "mean", "observed_year_months"),
        season_rows,
    )
    seasonal_chart(output / "seasonal_means.png", season_chart_groups, season_counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=PROJECT / "prepared")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "exploration")
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"새 출력 디렉터리를 지정해 주세요. {output}")
    try:
        use_korean_font(args.font)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".explore-", dir=output.parent) as temp:
            staging = Path(temp) / "exploration"
            staging.mkdir()
            explore(args.prepared_dir, staging)
            staging.rename(output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"시각화 오류. {error}\n")
    print(f"학습 시각화와 집계표 저장 위치 {output}")


if __name__ == "__main__":
    main()
