import numpy as np
import pandas as pd
import re
import numpy.typing as npt


def get_sum_edges(path_graph):
    graph = np.load(path_graph)
    packed_graph = graph["packed_graph"]
    nt = graph["nt"]
    unpacked_graph = np.unpackbits(packed_graph, axis=0)[:nt, :, :]
    sum_edges = np.sum(unpacked_graph, axis=(1, 2))
    return sum_edges, nt


def get_sum_triples(path_triple_data, path_df_filtered, nt):
    triple_data = np.load(path_triple_data)
    df_filtered = pd.read_excel(path_df_filtered, index_col=0)
    intensity = triple_data["triples_activity"]
    triples = triple_data["triples"]
    df_filtered2 = df_filtered[df_filtered["chi2_pval"] < 0.05]
    tr = from_str_to_array_triples(df_filtered2)
    indices = get_filtered_data_indices(triples, tr)
    filtered_intensity = intensity[indices, :]
    unpacked_filtered_intensity = np.unpackbits(filtered_intensity, axis=1)[:, :nt]
    sum_triples = np.sum(unpacked_filtered_intensity, axis=0)
    return sum_triples


def from_str_to_array_triples(df: pd.DataFrame) -> npt.ArrayLike:
    good_tr = df.index.values.tolist()
    tr = []
    for el in good_tr:
        tr0 = re.findall(r"\d+", el)
        tr1 = [int(i) for i in tr0]
        tr.append(tr1)
    tr = np.array(tr)
    return tr


def get_filtered_data_indices(
    triples: npt.ArrayLike, tr: npt.ArrayLike
) -> npt.ArrayLike:
    indices = np.zeros(len(tr), dtype="int32")
    for i in range(len(tr)):
        indices[i] = np.where((tr[i, None] == triples).all(axis=-1))[0]
    return indices
