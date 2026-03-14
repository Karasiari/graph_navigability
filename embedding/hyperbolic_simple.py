import pandas as pd
import polars as pl
import networkx as nx

import numpy as np
from itertools import combinations

#методы эмбеддингов:
#import umap
from sklearn.manifold import SpectralEmbedding, Isomap

import scipy as sp
import math


#coalescent embedding - предварительное взвешивание графа

def number_of_common_neighbors(g):
    """
    Optimized calculation of the number of common neighbors.
    """
    adj = nx.to_numpy_array(g)
    return adj @ adj

def external_degree(g):
    """
    Optimized calculation of external degrees.
    """
    adj = nx.to_numpy_array(g)
    degree = adj.sum(axis=1)  # Степень узлов
    cn = number_of_common_neighbors(g)  # Оптимизированные общие соседи
    ext_degree = (degree[:, None] - cn - 1) * adj
    return ext_degree.T

def RA1_weights(g):
    """
    Optimized RA1 weights calculation.
    """
    degree = np.array(g.degree())[:, 1].astype(float)
    di, dj = np.meshgrid(degree, degree)
    cn = number_of_common_neighbors(g)
    return (di + dj + di * dj) / (1 + cn)

def RA2_weights(g):
    """
    Optimized RA2 weights calculation.
    """
    ei = external_degree(g)
    cn = number_of_common_neighbors(g)
    return (ei + ei.T + ei * ei.T) / (1 + cn)

def EBC_weights(g):
    """
    Optimized edge betweenness centrality weights.
    """
    w = nx.edge_betweenness_centrality(g)
    edges = [(u, v, w_val) for (u, v), w_val in w.items()]
    _g = nx.Graph()
    _g.add_weighted_edges_from(edges)
    return nx.to_numpy_array(_g)


#Генерация angular coordinates

# Функции для координат
def CA_coords(coords):
    return np.array(coords) / np.linalg.norm(coords, ord=2, axis=-1, keepdims=True)

def EA_coords(coords):
    zero_angle_vec = np.array([1, 0]).reshape((1, 2))
    coords = CA_coords(coords)
    angles = np.arctan2(coords[:, 1], coords[:, 0])
    inds = np.argsort(angles)
    rescaled = np.arange(len(coords))[inds]
    rescaled = rescaled * 2 * np.pi / len(coords)
    return np.stack((np.cos(rescaled), np.sin(rescaled)), axis=1)

# Функция для оценки параметра степенного распределения
def get_pl_exponent(G):
    degree = np.array([d for _, d in G.degree()])
    results = powerlaw.Fit(degree, verbose=False)
    return results.power_law.alpha

# Радиальная координата
def radial_coord_deg(G, beta, zeta=1):
    degrees = np.array([d for _, d in G.degree()]).astype(float)
    inds = np.argsort(-degrees)
    return 2 / zeta * (beta * np.log(inds + 1) + (1 - beta) * np.log(len(degrees)))


# Обновлённая функция hc_embedding с Isomap по умолчанию

def hc_embedding(
    G,
    pre_weighting='RA1',
    embedding=None,
    angular_func=EA_coords,
    n_neighbors=15,
    min_dist=0.1,   # теперь НЕ используется, оставлен только для совместимости сигнатуры
    n_jobs=12
):
    """
    Computes a hyperbolic coalescent embedding of a given graph using Isomap for dimensionality reduction
    (если embedding не передан).

    Args:
        G: networkx.Graph
            The input graph.
        pre_weighting: str or callable
            Determines the features that are passed to the dimensionality reduction method.
            If a string, it should be the name of the pre-weighting function (e.g., 'RA1' or 'RA2').
            If a callable, it should be a function that takes a graph and returns a weight matrix.
        embedding: Object
            An embedding model that implements the fit_transform method (like sklearn.manifold.Isomap
            or sklearn.manifold.SpectralEmbedding).
        angular_func: callable
            A function that computes angular coordinates from the embedding.
        n_neighbors: int
            The number of neighbors for Isomap (only used if embedding is None).
        min_dist: float
            Kept only for backward compatibility; not used with Isomap.
        n_jobs: int
            Number of parallel jobs for Isomap (only used if embedding is None).

    Returns:
        pd.DataFrame:
            A dataframe mapping nodes to their (x, y) spatial coordinates in the embedded space.
    """
    # Если передана строка, получить функцию по имени
    weight_func = {'RA1': RA1_weights}.get(pre_weighting)
    if weight_func is None:
        raise ValueError(f"Pre-weighting function '{pre_weighting}' not found.")

    # Вычисление весов (матрица признаков / похожести)
    weights = weight_func(G)

    # Если модель эмбеддинга не указана, используем Isomap
    if embedding is None:
        embedding_model = Isomap(
            n_components=2,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs
        )
    else:
        embedding_model = embedding

    # Применяем модель эмбеддинга
    embedding = embedding_model.fit_transform(weights)

    # Угловые координаты
    angular_coords = angular_func(embedding)

    # Радиальные координаты
    gamma = get_pl_exponent(G)          # экспонента степенного распределения
    beta = 1.0 / (gamma - 1.0)

    radii = radial_coord_deg(G, beta)

    # Масштабируем угловые координаты радиусами
    coords = angular_coords * radii[..., None]

    # Сопоставляем узлы координатам
    node_ids = list(G.nodes)
    df = pd.DataFrame(coords, index=node_ids, columns=['x', 'y'])
    return df


# получаем матрицы для алгоритма из эмбеддинга

def hyperbolic_distance(x, y):
  r_x, angle_x = float(x[0]), float(x[1])
  r_y, angle_y = float(y[0]), float(y[1])
  delta_angle = math.pi - abs(math.pi - abs(angle_x - angle_y))

  cosh_dist = math.cosh(r_x) * math.cosh(r_y) - math.sinh(r_x) * math.sinh(r_y) * math.cos(delta_angle)
  dist = math.acosh(cosh_dist) if cosh_dist > 1 else 0.0
  return dist

def get_matrices(G, emb):
  x = emb['x'].values
  y = emb['y'].values
  r = np.sqrt(x**2 + y**2)
  angles = np.arctan2(y, x)
  coords = list(zip(r, angles))

  dist_matrix = [[hyperbolic_distance(x, y) for x in coords] for y in coords]
  adj_matrix = nx.to_numpy_array(G).astype(int).tolist()

  return adj_matrix, dist_matrix
