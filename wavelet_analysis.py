import numpy as np

def prepare_signal(signal, normalize=None):
    """Подготавливает сигнал и при необходимости нормализует его."""
    x = np.asarray(signal, dtype=float)

    if normalize == "zscore":
        x = (x - x.mean()) / (x.std() + 1e-12)
    elif normalize is not None:
        raise ValueError(f"Неизвестный способ нормализации: {normalize}")

    return x


def morlet_power_matrix(signal, frequencies, fs=20.0, w=6.0,
                        normalize=None):
    """Возвращает матрицу мощности |W(f, t)|²: (частоты, время)."""
    x = prepare_signal(signal, normalize=normalize)

    n = len(x)
    spectrum = np.fft.fft(x, 2 * n)
    omega = 2 * np.pi * np.fft.fftfreq(2 * n, 1 / fs) / fs

    power = np.empty((len(frequencies), n))

    for i, f0 in enumerate(frequencies):
        scale = w * fs / (2 * np.pi * f0)

        wavelet_fft = (
            np.sqrt(2 * np.pi * scale)
            * np.pi ** -0.25
            * np.exp(-0.5 * (scale * omega - w) ** 2)
        )

        power[i] = np.abs(
            np.fft.ifft(spectrum * wavelet_fft)[:n]
        ) ** 2

    return power


def morlet_mean_power(signal, frequencies, fs=20.0, w=6.0,
                      normalize=None):
    """Возвращает среднюю по времени мощность для каждой частоты."""
    power_matrix = morlet_power_matrix(
        signal,
        frequencies,
        fs=fs,
        w=w,
        normalize=normalize,
    )

    return power_matrix.mean(axis=1)