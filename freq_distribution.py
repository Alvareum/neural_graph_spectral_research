from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from scipy.optimize import curve_fit
from scipy.stats import nbinom
from scipy.optimize import minimize_scalar

# Глобально настраиваем академический стиль
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False
})

# Цвета из образца
C_NATIVE = "#1f5fa8"     # Синий для реальных/родных данных
C_SYNTHETIC = "#d9711a"  # Оранжевый для синтетических данных (по аналогии с LLM на образце)
C_FIT = "#666666"        # Серый для аппроксимаций


def plot_triple_distribution_sub(
        ax,
        data, 
        title=None,
        log_axis="both", 
        linear_bin=False, 
        need_curve_fit=False, 
        need_lognorm_fit=False,
        need_negative_binomial_fit=False,
        color=C_NATIVE,
        label="Triples distribution",
        fit_x_max=None
        ):
    """Функция для построения одного подграфика на переданных осях `ax`."""

    
    data_pos = data[data > 0]

    # Границы корзин
    if linear_bin:
        # bins = np.linspace(1, data_pos.max(), num=30)
        bins = np.arange(1, int(data_pos.max()) + 2, 1) 
    else:
        bins = np.geomspace(1, data_pos.max(), num=50).astype(int)

    bins = np.unique(bins)
    # Лог-бинирование и нормализация
    counts, bin_edges = np.histogram(data_pos, bins=bins)

    # Для всех бинов, кроме последнего, интервал [left, right) содержит (right - left) целых чисел.
    integer_counts_per_bin = np.diff(bin_edges)
    # Для самого последнего бина интервал [left, right] содержит (right - left + 1) целых чисел,
    # так как np.histogram включает туда крайнее правое значение x_max.
    integer_counts_per_bin[-1] += 1
    # Нормируем высоту гистограммы, чтобы она соответствовала средней дискретной плотности в бине
    # (Доля элементов в бине деленная на количество целых чисел в этом бине)
    density = counts / (len(data_pos) * integer_counts_per_bin)
    if linear_bin:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    else:
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:]) 
    valid = density > 0
    x_pure = bin_centers[valid]
    y_pure = density[valid]

    # Отображение фитов в стиле null-модели (тонкий серый пунктир)
    if need_curve_fit:
        # _ = add_powerlaw_fit_log(ax, x_pure, y_pure, color=C_FIT, linestyle='--')
        if fit_x_max is None:
            fit_x_max = int(data_pos.max())
        _ = add_zeta_fit_log(ax, data_pos, fit_x_max=fit_x_max, color=C_FIT, linestyle='--', label_prefix='Fitted Zeta')
    if need_lognorm_fit:
        _ = add_lognorm_fit(ax, x_pure, y_pure, color=C_FIT, linestyle='--')
    if need_negative_binomial_fit:
        add_negative_binomial_fit(ax, data_pos)

    # Масштабы осей
    if log_axis in ["both", "x"]:
        ax.set_xscale('log')
        ax.xaxis.set_minor_formatter(mt.NullFormatter())
    if log_axis in ["both", "y"]:
        ax.set_yscale('log')

    # Главный график (строгая линия без заливки)
    ax.plot(x_pure, y_pure, color=color, linewidth=1.6, label=label)

    # Оформление осей
    if title:
        ax.set_title(title, fontsize=8, fontweight='bold')
    ax.set_xlabel('Number of triples')
    ax.set_ylabel('Probability density')
    
    # Включаем строгую легенду без рамки
    if need_curve_fit or need_lognorm_fit or need_negative_binomial_fit:
        ax.legend(frameon=False, loc="best")


def generate_double_panel_plot(data_left, kwargs_left, data_right, kwargs_right, path_to_save=None):
    """Создает панель 1х2, передает индивидуальные параметры на каждый график и расставляет буквы a, b."""
    CM = 1 / 2.54  # перевод дюймов в сантиметры
    
    # Создаем двухпанельный рисунок в размерах образца (~16 см на 6.2 см)
    fig, ax = plt.subplots(1, 2, figsize=(16 * CM, 6.2 * CM))

    # Строим левый график (a) с его набором параметров
    plot_triple_distribution_sub(ax[0], data_left, color=C_NATIVE, **kwargs_left)
    
    # Строим правый график (b) с его набором параметров
    plot_triple_distribution_sub(ax[1], data_right, color=C_SYNTHETIC, **kwargs_right)

    # Автоматическое выравнивание элементов
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    # Добавление изящных букв a, b под графиками строго в стиле образца
    for a, letter in zip(ax, "ab"):
        fig.text(
            a.get_position().x0 - 0.06, 0.015, letter, 
            fontsize=10, style="italic", family="serif", 
            ha="left", va="bottom"
        )
        
    # Сохранение результатов
    if path_to_save:
        fig.savefig(path_to_save, bbox_inches='tight')
        raw_path = path_to_save.with_suffix(".png")
        fig.savefig(raw_path, dpi=300, bbox_inches='tight')
        
    plt.show()



def _prepare_zeta_sample(data, x_min=1, fit_x_max=300):
    """Проверяет данные и возвращает выборку в диапазоне фитирования."""
    data = np.asarray(data, dtype=float)

    if data.ndim != 1 or data.size == 0:
        raise ValueError("data должен быть непустым одномерным массивом.")

    if not np.all(np.isfinite(data)):
        raise ValueError("data содержит NaN или бесконечные значения.")

    if np.any(data <= 0) or not np.all(data == np.floor(data)):
        raise ValueError("Все значения data должны быть положительными целыми.")

    if not isinstance(x_min, (int, np.integer)) or x_min < 1:
        raise ValueError("x_min должен быть положительным целым числом.")

    if not isinstance(fit_x_max, (int, np.integer)) or fit_x_max < x_min:
        raise ValueError("fit_x_max должен быть целым числом >= x_min.")

    sample = data[(data >= x_min) & (data <= fit_x_max)]

    if len(sample) < 2 or np.all(sample == sample[0]):
        raise ValueError("Недостаточно разнообразных данных для фитирования.")

    return sample


def _zeta_negative_log_likelihood(s, sample, support):
    """Отрицательное логарифмическое правдоподобие усечённого Zeta-распределения."""
    normalization = np.sum(support ** (-s))

    return (
        len(sample) * np.log(normalization)
        + s * np.sum(np.log(sample))
    )


def _fit_zeta_parameter(sample, x_min=1, fit_x_max=300):
    """Оценивает параметр s методом максимального правдоподобия."""
    support = np.arange(x_min, fit_x_max + 1)

    result = minimize_scalar(
        _zeta_negative_log_likelihood,
        args=(sample, support),
        bounds=(1.000001, 20),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError("Не удалось оценить параметр Zeta.")

    return {
        "s": result.x,
        "log_likelihood": -result.fun,
    }


def _get_zeta_pmf(s, x_min=1, x_max=300, fit_x_max=300):
    """Вычисляет теоретическую PMF усечённого Zeta-распределения."""
    if x_max < x_min:
        raise ValueError("x_max должен быть >= x_min.")

    x = np.arange(x_min, x_max + 1)
    support = np.arange(x_min, fit_x_max + 1)

    normalization = np.sum(support ** (-s))
    probabilities = x.astype(float) ** (-s) / normalization

    return x, probabilities


def add_zeta_fit_log(
    ax,
    data,
    x_min=1,
    x_max=None,
    fit_x_max=300,
    color="#ff7f0e",
    linestyle="--",
    label_prefix="Fitted Zeta",
):
    """
    Оценивает параметр дискретного Zeta-распределения методом MLE
    и рисует теоретическую PMF на log-log графике.

    Распределение усечено на диапазоне [x_min, fit_x_max]:

        P(X=k) = k**(-s) / sum(j**(-s))

    Параметры
    ---------
    ax : matplotlib.axes.Axes
        Ось, на которой строится кривая.
    data : array-like
        Исходная выборка положительных целых чисел.
    x_min : int
        Минимальное значение выборки, включаемое в фитирование.
    x_max : int or None
        Верхняя граница отображения кривой.
        Если None, используется максимальное значение выборки.
    fit_x_max : int
        Верхняя граница усечённого распределения при фитировании.
    color : str
        Цвет кривой.
    linestyle : str
        Стиль линии.
    label_prefix : str
        Начало подписи в легенде.

    Returns
    -------
    dict
        Оценка параметра s, границы фитирования,
        размер выборки и log-likelihood.
    """
    sample = _prepare_zeta_sample(
        data,
        x_min=x_min,
        fit_x_max=fit_x_max,
    )

    if x_max is None:
        x_max = int(sample.max())

    x_max = int(x_max)

    fit_result = _fit_zeta_parameter(
        sample,
        x_min=x_min,
        fit_x_max=fit_x_max,
    )

    s_opt = fit_result["s"]

    x_line, p_line = _get_zeta_pmf(
        s=s_opt,
        x_min=x_min,
        x_max=x_max,
        fit_x_max=fit_x_max,
    )

    ax.plot(
        x_line,
        p_line,
        color=color,
        linestyle=linestyle,
        linewidth=1.6,
        label=rf"{label_prefix}: $s={s_opt:.3f}$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.legend(fontsize=10)

    return {
        "s": s_opt,
        "xmin": x_min,
        "fit_xmax": fit_x_max,
        "n": len(sample),
        "log_likelihood": fit_result["log_likelihood"],
    }


def add_negative_binomial_fit(ax, data):
    # 2. Estimate parameters using Method of Moments
    mean = np.mean(data)
    var = np.var(data)

    if var > mean:
        p_fit = mean / var
        r_fit = (mean ** 2) / (var - mean)
        print(f"Fitted Parameters:\n r (number of successes) = {r_fit:.2f}\n p (probability) = {p_fit:.2f}")
    else:
        raise ValueError("Variance must be greater than the mean to fit a Negative Binomial distribution.")

    x_range = np.arange(0, int(data.max()) + 1)
    pmf_fitted = nbinom.pmf(x_range, r_fit, p_fit)
    ax.plot(x_range, pmf_fitted, '--', color="#666666", lw=1.4, markersize=4, 
        label=f'Fitted NB Curve\n(r={r_fit:.1f}, p={p_fit:.2f})')
    ax.legend(fontsize=10)


def add_lognorm_fit(ax, x, y, x_min=None, x_max=None, color='magenta', linestyle='-.', label_prefix='Логнорм. фит'):
    """
    Подгоняет логнормальное распределение под сгруппированные данные (плотность) 
    и строит линию на графике.
    """
    # 1. Фильтруем данные (убираем нули и ограничиваем диапазон, если задан)
    mask = (x > 0) & (y > 0)
    if x_min is not None: mask &= (x >= x_min)
    if x_max is not None: mask &= (x <= x_max)
        
    x_fit = x[mask]
    y_fit = y[mask]
    
    if len(x_fit) < 3:
        print("Предупреждение: Слишком мало точек для логнормального фита.")
        return None

    # 2. Формула функции плотности логнормального распределения
    # s — стандартное отклонение логарифма (форма), scale — экспонента среднего (масштаб)
    # A — амплитуда (нужна, так как площадь под экспериментальной кривой может быть не строго 1)
    def lognorm_pdf(t, s, scale, A):
        # Защита от деления на ноль и некорректных параметров
        if s <= 0 or scale <= 0:
            return np.zeros_like(t)
        # Стандартная формула scipy.stats.lognorm.pdf
        return A * (1.0 / (t * s * np.sqrt(2 * np.pi))) * np.exp(-((np.log(t) - np.log(scale))**2) / (2 * s**2))

    try:
        # 3. Начальное приближение (p0) и границы (bounds), чтобы алгоритм не ушел в ошибку
        # [s, scale, A]
        p0 = [1.0, np.median(x_fit), y_fit.max()]
        bounds = ((0.01, 0.01, 0.0), (10.0, x_fit.max() * 10, y_fit.max() * 100))
        
        popt, _ = curve_fit(lognorm_pdf, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
        s_opt, scale_opt, A_opt = popt
        
        # 4. Генерируем точки для плавной линии
        x_line = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 200)
        y_line = lognorm_pdf(x_line, s_opt, scale_opt, A_opt)
        
        # 5. Отрисовка
        label = rf"{label_prefix}: $\sigma={s_opt:.2f}$, $\mu(scale)={scale_opt:.1f}$"
        ax.plot(x_line, y_line, color=color, linestyle=linestyle, linewidth=1.4, label=label)
        ax.legend(fontsize=10)
        
        return {"s": s_opt, "scale": scale_opt, "amplitude": A_opt}
        
    except Exception as e:
        print(f"Не удалось построить логнормальный фит: {e}")
        return None