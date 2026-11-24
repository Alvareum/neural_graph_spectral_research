import numpy as np
import numpy.typing as npt


def make_dynamic_network_spikes(
    spikes: npt.ArrayLike,
    lag_window: int = 40,
    last_spike: bool = False,
    is_flexible: bool = True,
) -> npt.ArrayLike:
    n = spikes.shape[1]
    nt = spikes.shape[0]
    dynamic_network = np.zeros([n, n, nt], dtype=bool)
    direct = np.zeros(nt, dtype=bool)
    reverse = np.zeros(nt, dtype=bool)
    mask = np.zeros(nt, dtype=bool)
    for id_cell_1 in range(n):
        for id_cell_2 in range(n):
            if id_cell_1 == id_cell_2:
                continue
            if id_cell_1 > id_cell_2:
                continue
            starts_1 = np.argwhere(spikes[:, id_cell_1] > 0).flatten()
            starts_2 = np.argwhere(spikes[:, id_cell_2] > 0).flatten()
            direct = find_lagged_events(
                starts_1,
                starts_2,
                nt,
                direct,
                mask,
                lag_window,
                last_spike,
                is_flexible,
            )
            dynamic_network[id_cell_1, id_cell_2, :] = direct
            reverse = find_lagged_events(
                starts_2,
                starts_1,
                nt,
                reverse,
                mask,
                lag_window,
                last_spike,
                is_flexible,
            )
            dynamic_network[id_cell_2, id_cell_1, :] = reverse
    return dynamic_network


def find_lagged_events(
    starts_1: npt.ArrayLike,
    starts_2: npt.ArrayLike,
    nt: int,
    direct: npt.ArrayLike,
    mask: npt.ArrayLike,
    lag_window: int = 40,
    last_spike: bool = False,
    is_flexible: bool = True,
) -> npt.ArrayLike:
    direct[:] = False
    mask[:] = False
    # starts_2_args = np.argwhere(starts_2).flatten()
    mask[starts_2] = 1
    for i in range(len(starts_1)):
        be = starts_1[i]
        en = min(
            [nt, starts_1[i] + lag_window + 1]
        )  # +1 чтобы было 41, как в матлабе, так как последнее число в индексации не входит
        tmp_arr = mask[be:en]
        sp = np.argwhere(tmp_arr).flatten()
        if not is_flexible and len(sp) > 0:
            direct[be:en] = 1
        else:
            if last_spike and len(sp) > 0:
                en = (
                    be + int(sp[-1]) + 1
                )  # +1 чтобы последнее число входило как горение ребра
                direct[be:en] = 1
            elif len(sp) > 0:
                en = be + int(sp[0]) + 1
                direct[be:en] = 1
    return direct
