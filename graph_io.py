import numpy as np
from pathlib import Path

# Graph preparing


def pack_graph(graph):
    """
    make uint8 bits packed graph
    graph : bool array with shape (nt, n, n)
    """
    packed_graph = np.packbits(graph, axis=0)
    return packed_graph


def make_undirected_graph(graph):
    """
    if edge exist in any direction then it will be in undirected graph
    graph : bool array with shape (nt, n, n) or packed graph
    """
    graph = graph | graph.swapaxes(1, 2)
    return graph


def prepare_graph(graph):
    """
    graph : bool array with shape (n, n, nt)
    """
    graph = graph.swapaxes(0, 2)
    packed_graph = pack_graph(graph)
    undirected_packed_graph = make_undirected_graph(packed_graph)
    nt = graph.shape[0]
    return undirected_packed_graph, packed_graph, nt


def unpack_graph(packed_graph, nt):
    graph = np.unpackbits(packed_graph, axis=0)
    graph = graph[:nt, :, :]
    return graph


# Graph save/load


def build_graph_path(path_to_save, filename, method, is_packed=True):
    names = ["graph", "packed" if is_packed else "unpacked", f"{filename}", f"{method}"]
    name = "_".join(names)
    return Path(path_to_save) / f"{name}.npz"


def save_graph(path_graph, undirected_packed_graph, packed_graph, nt):
    np.savez_compressed(
        path_graph,
        undirected_packed_graph=undirected_packed_graph,
        packed_graph=packed_graph,
        nt=nt,
    )


def load_graph(path_graph, is_directed=False, only_info=False):
    file = np.load(path_graph)
    nt = file["nt"]
    if only_info:
        return nt
    if is_directed:
        packed_graph = file["packed_graph"]
    else:
        packed_graph = file["undirected_packed_graph"]
    return packed_graph, nt


# End of graph preparing