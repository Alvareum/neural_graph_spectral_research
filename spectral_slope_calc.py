"""Расчёт спектральных наклонов и построение рисунка 4."""

import json
from pathlib import Path

import numpy as np
from wavelet_analysis import morlet_power_matrix
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory


def load_mean_power(cwt_path, sum_path, kind, frequencies_default, fs=20.0, w=6.0):
    """Загружает сохранённую CWT-мощность или рассчитывает её из исходного ряда."""
    if cwt_path is not None and Path(cwt_path).exists():
        with np.load(cwt_path) as data:
            power = data["power_matrix"]
            freqs = data["frequencies"] if "frequencies" in data.files else frequencies_default
        source = "saved power_matrix"
    elif sum_path is not None and Path(sum_path).exists():
        with np.load(sum_path) as data:
            signal = data[f"sum_{kind}"]
        freqs = frequencies_default
        power = morlet_power_matrix(signal, freqs, fs=fs, w=w)
        source = f"sum_{kind} -> CWT"
    else:
        raise FileNotFoundError(f"Не найден ни CWT-файл {cwt_path}, ни файл ряда {sum_path}")

    if power.shape[0] != len(freqs):
        power = power.T
    if power.shape[0] != len(freqs):
        raise ValueError(f"Размер power_matrix {power.shape} не согласуется с {len(freqs)} частотами")
    return np.asarray(freqs, dtype=float), np.asarray(power).mean(axis=1), source


def to_log_grid(freqs, power, f_range, n_points):
    """Интерполирует спектр на равномерную логарифмическую сетку."""
    mask = (freqs >= f_range[0]) & (freqs <= f_range[1]) & (power > 0)
    if mask.sum() < 5:
        raise ValueError(f"Недостаточно положительных точек в диапазоне {f_range}")
    log_freq = np.log10(freqs[mask])
    log_power = np.log10(power[mask])
    order = np.argsort(log_freq)
    grid = np.linspace(log_freq[order][0], log_freq[order][-1], n_points)
    return grid, np.interp(grid, log_freq[order], log_power[order])


def local_slope(log_freq, log_power, window_decades=0.3):
    """Локальный beta(f) = -d log(P) / d log(f)."""
    beta = np.full(len(log_freq), np.nan)
    for i, center in enumerate(log_freq):
        half = window_decades / 2
        if center - half < log_freq[0] or center + half > log_freq[-1]:
            continue
        mask = np.abs(log_freq - center) <= half
        if mask.sum() >= 5:
            beta[i] = -np.polyfit(log_freq[mask], log_power[mask], 1)[0]
    return 10 ** log_freq, beta


def slope_metrics(freqs, mean_power, f_range, band, n_log, window_decades):
    """Возвращает максимум локального beta, частоту максимума и beta в заданной полосе."""
    log_freq, log_power = to_log_grid(freqs, mean_power, f_range, n_log)
    local_freqs, beta = local_slope(log_freq, log_power, window_decades)
    if not np.isfinite(beta).any():
        raise ValueError("Не удалось вычислить локальные наклоны: проверьте диапазон и окно")
    max_index = int(np.nanargmax(beta))
    band_mask = (log_freq >= np.log10(band[0])) & (log_freq <= np.log10(band[1]))
    if band_mask.sum() < 2:
        raise ValueError(f"В полосе {band} недостаточно точек для оценки наклона")
    band_beta = -np.polyfit(log_freq[band_mask], log_power[band_mask], 1)[0]
    return float(beta[max_index]), float(local_freqs[max_index]), float(band_beta)


def compute_all(mice, days, kinds, cwt_dir, cwt_pattern, sum_dir, sum_pattern,
                llm_cwt, llm_sum, frequencies_default, cache_file,
                f_range, band, n_log, window_decades, fs=20.0, w=6.0,
                recompute=False):
    """Рассчитывает метрики для всех мышей, дней и LLM; использует JSON-кэш."""
    cache_file = Path(cache_file)
    if cache_file.exists() and not recompute:
        print(f"Результаты загружены из кэша: {cache_file}")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    results = {kind: {} for kind in kinds}
    for kind in kinds:
        for mouse in mice:
            for day in days:
                record = f"{mouse}_{day}"
                video = f"CA1_{record}"
                cwt_path = Path(cwt_dir) / cwt_pattern.format(kind=kind, video=video) if cwt_dir else None
                sum_path = Path(sum_dir) / sum_pattern.format(video=video) if sum_dir else None
                freqs, mean_power, source = load_mean_power(
                    cwt_path, sum_path, kind, frequencies_default, fs, w
                )
                metrics = slope_metrics(freqs, mean_power, f_range, band, n_log, window_decades)
                results[kind][record] = dict(
                    max_beta=metrics[0], f_max=metrics[1],
                    band_beta=metrics[2], source=source
                )

        freqs, mean_power, source = load_mean_power(
            llm_cwt[kind], llm_sum, kind, frequencies_default, fs, w
        )
        metrics = slope_metrics(freqs, mean_power, f_range, band, n_log, window_decades)
        results[kind]["LLM"] = dict(
            max_beta=metrics[0], f_max=metrics[1],
            band_beta=metrics[2], source=source
        )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def x_positions(mice, days, group_gap):
    """Координаты записей: группы мышей разделены дополнительным промежутком."""
    return {
        f"{mouse}_{day}": i * (len(days) + group_gap) + j
        for i, mouse in enumerate(mice) for j, day in enumerate(days)
    }


def draw_panel(ax, results, kind, title, mice, days, group_gap,
               mouse_color, llm_color, reference_color):
    """Рисует одну панель: edges или triples."""
    values = {
        record: (item["max_beta"], item["band_beta"])
        for record, item in results[kind].items()
    }
    positions = x_positions(mice, days, group_gap)
    group_width = len(days) - 1

    for i, mouse in enumerate(mice):
        x0 = i * (len(days) + group_gap)
        if i % 2 == 0:
            ax.axvspan(x0 - 0.5, x0 + group_width + 0.5,
                       color="0.93", zorder=0, lw=0)
        keys = [f"{mouse}_{day}" for day in days]
        xs = [positions[key] for key in keys]
        maximum = [values[key][0] for key in keys]
        average = [values[key][1] for key in keys]
        ax.plot(xs, maximum, color=mouse_color, lw=0.8, alpha=0.6, zorder=2)
        ax.plot(xs, average, color=mouse_color, lw=0.8, alpha=0.6, ls="--", zorder=2)
        ax.scatter(xs, maximum, s=20, color=mouse_color, zorder=3)
        ax.scatter(xs, average, s=20, facecolor="white", edgecolor=mouse_color, zorder=3)

    ax.axhline(results[kind]["LLM"]["max_beta"], color=llm_color, lw=1.4, zorder=1)
    ax.axhline(2.0, color=reference_color, lw=0.8, ls=":", zorder=1)
    ax.set_xticks([positions[f"{m}_{d}"] for m in mice for d in days])
    ax.set_xticklabels([day.replace("D", "") for m in mice for day in days])
    ax.tick_params(axis="x", length=2, pad=2)

    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for i, mouse in enumerate(mice):
        center = i * (len(days) + group_gap) + group_width / 2
        ax.text(center, -0.13, f"CA1_{mouse}", transform=transform,
                ha="center", va="top", fontsize=7.5)
    ax.text(1.0, -0.035, "day", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color="0.3")
    ax.set_xlim(-0.7, (len(mice) - 1) * (len(days) + group_gap) + group_width + 0.7)
    ax.set_title(title)


def plot_slopes(results, mice, days, kinds, output_pdf, output_png,
                *, figsize_cm=(16, 8), group_gap=1.0,
                mouse_color="#1f5fa8", llm_color="#d9711a",
                reference_color="#666666", reference_beta=2.0,
                y_min=0.0, y_max_min=5.8, dpi=300, show=False):
    """Собирает две панели и сохраняет рисунок в PDF и PNG."""
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    })
    cm = 1 / 2.54
    fig = plt.figure(figsize=(figsize_cm[0] * cm, figsize_cm[1] * cm))
    grid = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 0.24, 0.10],
        left=0.075, right=0.97, top=0.92, bottom=0.03,
        wspace=0.12, hspace=0.05
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], sharey=ax_a)
    draw_panel(ax_a, results, kinds[0], "Edges", mice, days, group_gap,
               mouse_color, llm_color, reference_color)
    draw_panel(ax_b, results, kinds[1], "Triples", mice, days, group_gap,
               mouse_color, llm_color, reference_color)
    ax_a.set_ylabel(r"Spectral slope $\beta$")
    top = max(v["max_beta"] for kind in kinds for v in results[kind].values())
    ax_a.set_ylim(y_min, max(y_max_min, top * 1.08))
    plt.setp(ax_b.get_yticklabels(), visible=False)

    for col, letter in enumerate("ab"):
        label_ax = fig.add_subplot(grid[1, col])
        label_ax.axis("off")
        label_ax.text(-0.06, 0.0, letter, transform=label_ax.transAxes,
                      fontsize=10, style="italic", family="serif",
                      ha="left", va="bottom")

    handles = [
        Line2D([], [], color=mouse_color, marker="o", ms=4.5, lw=0.8,
               label=r"Maximum local $\beta$"),
        Line2D([], [], color=mouse_color, marker="o", ms=4.5, mfc="white",
               lw=0.8, ls="--", label=r"Mean $\beta$, 0.3–1 Hz"),
        Line2D([], [], color=llm_color, lw=1.4, label=r"LLM: maximum local $\beta$"),
        Line2D([], [], color=reference_color, lw=0.8, ls=":",
               label=fr"$\beta = {reference_beta:g}$"),
    ]
    legend_ax = fig.add_subplot(grid[2, :])
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center", ncol=4,
                     frameon=False, handlelength=2.2, columnspacing=1.8)

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Сохранено: {output_pdf} и {output_png}")
