import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from pathlib2 import Path

def plot_wavelet_comparison(
    results,
    frequencies,
    *,
    bursts=None,
    dips=None,
    half_width_sec=5,
    power_limits=(-20, 50),
    frequency_limits=(0, 10),
    frequency_ticks=(0, 5, 10),
    cmap="magma",
    figsize=(12, 6.5),
    save_dir=None,
    figure_name="wavelet_comparison",
    highlight_experimental_only=True,
):
    """
    Строит сравнительный рисунок сигналов и локальных
    вейвлетных спектров.

    Parameters
    ----------
    results : list of dict
        Данные сигналов и спектров.

    frequencies : array
        Частоты CWT.

    bursts, dips : list or None
        Времена всплесков и провалов, секунды.

    half_width_sec : float
        Полуширина интервалов выделения.

    power_limits : tuple
        Минимум и максимум мощности в dB.

    frequency_limits : tuple
        Диапазон частот.

    frequency_ticks : tuple
        Отметки частотной оси.

    cmap : str
        Цветовая карта.

    figsize : tuple
        Размер рисунка.

    save_dir : Path or str or None
        Папка сохранения.

    figure_name : str
        Имя файлов рисунка.

    highlight_experimental_only : bool
        Выделять события только на первой панели.
    """

    bursts = [] if bursts is None else bursts
    dips = [] if dips is None else dips

    power_vmin, power_vmax = power_limits
    freq_min, freq_max = frequency_limits

    # Создание макета
    fig, axes_signal, axes_cwt, cbar_axes = create_figure(
        figsize=figsize
    )

    meshes = []

    for i, result in enumerate(results):

        show_events = (
            not highlight_experimental_only or i == 0
        )

        # Верхний ряд: сигналы
        plot_signal(
            axes_signal[i],
            result,
            bursts=bursts,
            dips=dips,
            half_width_sec=half_width_sec,
            highlight_events=show_events
        )

        # Нижний ряд: вейвлетные спектры
        mesh = plot_cwt_spectrum(
            axes_cwt[i],
            result,
            frequencies,
            power_vmin=power_vmin,
            power_vmax=power_vmax,
            freq_min=freq_min,
            freq_max=freq_max,
            freq_ticks=frequency_ticks,
            cmap=cmap,
            bursts=bursts,
            dips=dips,
            half_width=half_width_sec,
            show_events=show_events
        )

        meshes.append(mesh)

    # Цветовые шкалы
    add_colorbars(
        fig,
        cbar_axes,
        meshes,
        power_min=power_vmin,
        power_max=power_vmax,
        label="Power, dB",
        orientation="horizontal"
    )

    # Обозначения панелей
    add_panel_labels(fig)

    # Сохранение
    if save_dir is not None:

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_dir / f"{figure_name}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white"
        )

        fig.savefig(
            save_dir / f"{figure_name}.pdf",
            dpi=300,
            bbox_inches="tight",
            facecolor="white"
        )

    plt.show()

    return fig


def create_figure(figsize=(12, 6.5)):
    """
    Создаёт макет рисунка с двумя колонками:
    сигналы, спектрограммы и цветовые шкалы.
    """

    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[0.85, 2.0, 0.25],
        hspace=0.38,
        wspace=0.25,
        left=0.08,
        right=0.98,
        top=0.94,
        bottom=0.11
    )

    axes_signal = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1])
    ]

    axes_cwt = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1])
    ]

    cbar_axes = [
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1])
    ]

    return fig, axes_signal, axes_cwt, cbar_axes

def plot_signal(
    ax,
    result,
    highlight_events=False,
    *,
    signal_color="0.40",
    signal_linewidth=0.55,
    signal_ylabel="Tr(t)",
    bursts=None,
    dips=None,
    half_width_sec=5
):
    """
    Строит исходный сигнал числа троек мыши или LLM.
    """

    times = result["times"]
    signal = result["signal"]

    # Выделение событий
    if highlight_events and (bursts is not None or dips is not None):
        highlight_signal_events(ax, times, bursts=bursts, dips=dips, half_width=half_width_sec)

    # Исходный сигнал
    ax.plot(
        times,
        signal,
        color=signal_color,
        linewidth=signal_linewidth,
        zorder=2
    )

    ax.set_title(
        result["title"],
        fontsize=15,
        pad=10
    )

    ax.set_ylabel(
        signal_ylabel,
        fontsize=12
    )

    ax.set_xlim(times[0], times[-1])

    style_axis(ax)

    return ax

def highlight_signal_events(
    ax,
    times,
    *,
    bursts=None,
    dips=None,
    half_width=5,
    alpha=0.3,
    dip_color="darkviolet",
    burst_color="orangered"
):
    """
    Выделяет всплески и провалы на графике сигнала.
    """

    t_min = times[0]
    t_max = times[-1]

    # Провалы
    if dips is not None:
        for t in dips:
            x0 = max(t_min, t - half_width)
            x1 = min(t_max, t + half_width)

            ax.axvspan(
                x0, x1,
                color=dip_color,
                alpha=alpha,
                linewidth=0,
                zorder=1
            )

    # Всплески
    if bursts is not None:
        for t in bursts:
            x0 = max(t_min, t - half_width)
            x1 = min(t_max, t + half_width)

            ax.axvspan(
                x0, x1,
                color=burst_color,
                alpha=alpha,
                linewidth=0,
                zorder=1
            )

def plot_cwt_spectrum(
    ax,
    result,
    frequencies,
    show_events=False,
    *,
    power_vmin=-20,
    power_vmax=50,
    cmap="magma",
    spectrum_xlabel="Time, s",
    spectrum_ylabel="Frequency, Hz",
    freq_min=0,
    freq_max=10,
    freq_ticks=(0, 5, 10),
    bursts=None,
    dips=None,
    half_width=5
):
    """
    Строит локальный вейвлетный спектр.
    """

    times = result["times"]
    power_db = result["power_db"]

    mesh = ax.pcolormesh(
        times,
        frequencies,
        power_db,
        shading="auto",
        cmap=cmap,
        vmin=power_vmin,
        vmax=power_vmax,
        rasterized=True
    )

    ax.set_xlabel(
        spectrum_xlabel,
        fontsize=12
    )

    ax.set_ylabel(
        spectrum_ylabel,
        fontsize=12
    )

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(freq_min, freq_max)
    ax.set_yticks(freq_ticks)

    style_axis(ax)

    # Скобки над экспериментальной спектрограммой
    if show_events:
        add_event_brackets(ax, times, bursts=bursts, dips=dips, half_width=half_width)

    return mesh

def style_axis(ax, labelsize=11):
    """
    Единое оформление осей.
    """

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        direction="out",
        length=5,
        width=1.1,
        labelsize=labelsize
    )

def add_event_brackets(
    ax,
    times,
    bursts=None,
    dips=None,
    half_width=5
):
    """
    Добавляет квадратные скобки для всплесков и провалов.
    """

    t_min = times[0]
    t_max = times[-1]

    # Провалы
    if dips is not None:
        for t in dips:

            x0 = max(t_min, t - half_width)
            x1 = min(t_max, t + half_width)

            add_square_bracket(
                ax,
                x0,
                x1 + min(5, t_max - x1),
                label="−",
                color="#303030",
                y=1.035
            )

    # Всплески
    if bursts is not None:
        for t in bursts:

            x0 = max(t_min, t - half_width)
            x1 = min(t_max, t + half_width)

            add_square_bracket(
                ax,
                x0,
                x1 + min(5, t_max - x1),
                label="*",
                color="#303030",
                y=1.035
            )

def add_square_bracket(
    ax, 
    t_start, 
    t_end, 
    y=1.08,
    label=None, 
    color="black", 
    lw=1.5,
    fontsize=11
):
    """
    t_start, t_end — границы интервала в координатах оси X.
    y — высота над верхней границей оси в координатах Axes.
    """
    trans = ax.get_xaxis_transform()

    # Вертикальные края и горизонтальная перекладина
    h = 0.035

    ax.plot(
        [t_start, t_start, t_end, t_end],
        [y - h, y, y, y - h],
        transform=trans,
        color=color,
        linewidth=lw,
        clip_on=False
    )

    if label is not None:
        ax.text(
            (t_start + t_end) / 2,
            y + 0.015,
            label,
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            clip_on=False
        )

def add_colorbars(fig, 
                  cbar_axes, 
                  meshes,
                  *,
                  power_min=-20,
                  power_max=50,
                  label="Power, dB",
                  orientation="horizontal",
                  fontsize=12,
                  labelpad=7,
                  labelsize=11,
                  linewidth=0.8
                  ):
    """
    Добавляет горизонтальные цветовые шкалы.
    """

    colorbars = []

    for cax, mesh in zip(cbar_axes, meshes):

        cbar = fig.colorbar(
            mesh,
            cax=cax,
            orientation=orientation
        )

        cbar.set_label(
            label,
            fontsize=fontsize,
            labelpad=labelpad
        )

        cbar.set_ticks(
            np.arange(power_min, power_max + 1, 10)
        )

        cbar.ax.tick_params(labelsize=labelsize)
        cbar.outline.set_linewidth(linewidth)

        colorbars.append(cbar)

    return colorbars

def add_panel_labels(fig):
    """
    Добавляет обозначения панелей a и b.
    """

    fig.text(
        0.015, 0.025, "a",
        fontsize=23,
        fontfamily="serif",
        fontstyle="italic"
    )

    fig.text(
        0.515, 0.025, "b",
        fontsize=23,
        fontfamily="serif",
        fontstyle="italic"
    )
