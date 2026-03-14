from typing import List, Tuple

Matrix = List[List[int | float]]
Levels = Tuple[Tuple[int, List[int]]]

# вспомогательная для расчета betweenness функция для разбивки графа на уровни по метрике

def get_disc_levels(
    distances_from_t: List[float], 
    t: int
) -> Levels:
    """
    Рассчитывает разбиение графа на уровни для выделенной вершины по расстояниям до нее
    Input:
          distances_from_t - список расстояний до вершины t, относительно которой считаем уровни
          t - выделенная вершина, для которой считаем уровни
    Output:
          Разбиение вершин на уровни по расстоянию до t как кортеж кортежей вида (номер уровня, список вершин)
    """
    vertex_distances = [(dist, vertex) for vertex, dist in enumerate(distances_from_t)]

    vertex_distances.sort(key=lambda x: x[0])

    disc = []
    current_level_num = 0
    current_level = []
    current_distance = None

    for dist, vertex in vertex_distances:
        if dist != current_distance:
            if current_level:
                disc.append((current_level_num, current_level))
                current_level_num += 1
            current_level = [vertex]
            current_distance = dist
        else:
            current_level.append(vertex)

    if current_level:
        disc.append((current_level_num, current_level))

    return disc


# основная функция для вычисления betweenness - сложность |V|*|E|

def calculate_navigational_betweenness(
  adj_matrix: Matrix, 
  dist_matrix: Matrix
) -> List[float]:
  """
  Функция с алгоритмом вычисления navigational betweenness (Brandes like) для всех вершин
  Input: 
        adj_matrix - матрица смежности графа
        dist_matrix - матрица расстояний графа
  Output:
        Значения navigational betweenness для вершин-индексов как список
  """
  n = len(adj_matrix)
  neighbors = []
  for v in range(n):
    neighbors_v = [i for i in range(n) if adj_matrix[v][i] == 1]
    neighbors.append(neighbors_v)
  delta = [[None for i in range(n)] for j in range(n)]

  for t in range(n):
    # инициализация рабочих списков
    con = [0] * n
    n_t, p_t = [[] for i in range(n)], [[] for i in range(n)]
    sigma_t = [0] * n

    # 1) вычисление уровней
    disc = get_disc_levels(dist_matrix, t, n)

    # инициализация в t
    con[t] = 1
    sigma_t[t] = None

    # 2) BFS в одну сторону с получением списков n_t, p_t
    for level in disc:
      if level[0] != 0:
        for vertex in level[1]:
          min_neighbors_dist = None
          for neighbor in neighbors[vertex]:
            if (min_neighbors_dist is None) or (dist_matrix[t][neighbor] < min_neighbors_dist):
              min_neighbors_dist = dist_matrix[t][neighbor]

          for neighbor in neighbors[vertex]:
            if (con[neighbor] == 1) and (dist_matrix[t][neighbor] == min_neighbors_dist) and (min_neighbors_dist < dist_matrix[t][vertex]):
              n_t[vertex].append(neighbor)
              if sigma_t[neighbor] is None:
                sigma_t[vertex] = 1
              else:
                sigma_t[vertex] += sigma_t[neighbor]
              con[vertex] = 1
              p_t[neighbor].append(vertex)

    # 3) BFS в обратную с получением delta
    level_max_num = disc[-1][0]
    level_max = disc[level_max_num][1]
    for vertex in level_max:
      delta[t][vertex] = 0

    level_num = level_max_num - 1
    while level_num > 0:
      level = disc[level_num][1]
      for vertex in level:
        delta[t][vertex] = 0
        for u in p_t[vertex]:
          delta[t][vertex] += (sigma_t[vertex]/sigma_t[u]) * (delta[t][u] + 1)
      level_num -= 1

  # подсчет navigational betweenness
  B_nav = [0] * n
  for vertex in range(n):
    for t in range(n):
      if t != vertex:
        B_nav[vertex] += delta[t][vertex]

  return B_nav
