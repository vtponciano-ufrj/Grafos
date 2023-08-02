https://colab.research.google.com/drive/1a1KmLLaNf6v8nTSxPS0PKrDzYkej-GXB


import networkx as nx
import random
import os

def threshold(v, G):
      return 2

def contaminar(contaminado, vcontaminante, vcontaminado):
    if vcontaminado in contaminado:
        if contaminado[vcontaminado] is None:
            return False
        if len(contaminado[vcontaminado]) < 2 and vcontaminante not in contaminado[vcontaminado]:
            contaminado[vcontaminado].add(vcontaminante)
            if len(contaminado[vcontaminado]) == 2:
                return True
    else:
        contaminado[vcontaminado] = set()
        contaminado[vcontaminado].add(vcontaminante)
        if len(contaminado[vcontaminado]) == 2:
            return True
    return False

def init_contaminado(fecho):
    contaminado = {}
    for v in fecho:
        contaminado[v] = None
    return contaminado

def get_novo_fecho(fecho, graph, contaminado):
    novo_fecho = set()
    for v in fecho:
        if v in graph:
            for w in graph[v]:
                novo_contaminado = contaminar(contaminado, v, w)
                if novo_contaminado:
                    novo_fecho.add(w)
    return novo_fecho





def MTS(G, t):
    V = set()
    L = set()
    U = set(G.nodes())

    k = {v: t[v] for v in G}
    delta = {v: G.degree(v) for v in G}

    for v in U:
        if G.degree(v) == 1:
            V.add(v)

    while U:
        if any(k[v] == 0 for v in U):
            eligible_nodes = [v for v in U if k[v] == 0]
            v = random.choice(eligible_nodes)

            for u in set(G.neighbors(v)) & U:
                k[u] = max(k[u] - 1, 0)

                if v not in L:
                    delta[u] -= 1

            U.remove(v)

        else:
            candidates_1 = [v for v in U - L if delta[v] < k[v]]

            if candidates_1:
                v = random.choice(candidates_1)

                V.add(v)

                for u in set(G.neighbors(v)) & U:
                    k[u] -= 1
                    delta[u] -= 1

                U.remove(v)

            else:
                v = None
                candidates_2 = [u for u in U - L]

                if candidates_2:
                    min_candidate = min(candidates_2, key=lambda u: (k[u], delta[u] * (delta[u] + 1)))
                    v = random.choice([min_candidate])

                for u in set(G.neighbors(v)) & U:
                    delta[u] -= 1

                L.add(v)

    fecho = set(V)  # Initialize fecho with the remaining vertices in V
    print("Tamanho do Grafo:", len(G.nodes()))
    print("Tamanho do fecho inicial:", len(fecho))

    # Some code related to init_contaminado and get_novo_fecho is missing, please define or import them.
    contaminado = init_contaminado(fecho)
    time = 0
    while True:
        G_contamination = G
        novo_fecho = get_novo_fecho(fecho, G, contaminado)
        fecho.update(novo_fecho)  # Add the new vertices to fecho
        contaminated_vertices = novo_fecho  # Use contaminado instead of novo_fecho
        for v in contaminated_vertices:
            G_contamination.nodes[v]['contaminated_at_time'] = time
        time += 1

        if len(novo_fecho) == 0:
            break


    # Write the graph to a GEXF file
    nx.write_gexf(G_contamination, "contamination_graph.gexf")

    print("Tamanho dos Gerados:", len(fecho))
    print("Tamanho do fecho final: {}, Tempo final: {}".format(len(fecho), time))

    return V





import networkx as nx
import random
import os

def threshold(v, G):
      return 2

def init_contaminado(fecho):
    contaminado = {}
    for v in fecho:
        contaminado[v] = None
    return contaminado

def get_novo_fecho(fecho, graph, contaminado):
    novo_fecho = set()
    for v in fecho:
        if v in graph:
            for w in graph[v]:
                novo_contaminado = contaminar(contaminado, v, w)
                if novo_contaminado:
                    novo_fecho.add(w)
    return novo_fecho

def contaminar(contaminado, vcontaminante, vcontaminado):
    if vcontaminado in contaminado:
        if contaminado[vcontaminado] is None:
            return False
        if len(contaminado[vcontaminado]) < 2 and vcontaminante not in contaminado[vcontaminado]:
            contaminado[vcontaminado].add(vcontaminante)
            if len(contaminado[vcontaminado]) == 2:
                return True
    else:
        contaminado[vcontaminado] = set()
        contaminado[vcontaminado].add(vcontaminante)
        if len(contaminado[vcontaminado]) == 2:
            return True
    return False

def tip_decomp_P3_Random(threshold, G):
    V = set(G.nodes())
    dist = {}
    activated_nodes = set()

    # Incluir todos os vértices no conjunto inicial V'
    for vi in V:
        ki = threshold(vi, G)
        dist[vi] = G.degree(vi) - ki
        if G.degree(vi) == 1:
            dist[vi] = float('inf')

    flag = True
    while flag:
        min_value = min(dist.values())
        min_nodes = [node for node, value in dist.items() if value == min_value]
        vi = random.choice(min_nodes)



        if dist[vi] == float('inf'):
            flag = False
        else:
            V.remove(vi)
            del dist[vi]  # Remover o nó do dicionário dist
            neighbors = list(G.neighbors(vi))
            for vj in neighbors:
                if vj in dist:
                    if dist[vj] > 0:
                        dist[vj] -= 1
                    else:
                        dist[vj] = float('inf')


    fecho = set(V)  # Inicializar fecho com os vértices restantes em V
    print("Tamanho do Grafo:", len(G.nodes()))
    print("Tamanho do fecho inicial:", len(fecho))

    print("Tamanho do Grafo:", len(G.nodes()))
    print("Tamanho do fecho inicial:", len(fecho))

    # Some code related to init_contaminado and get_novo_fecho is missing, please define or import them.
    contaminado = init_contaminado(fecho)
    time = 0
    while True:
        G_contamination = G
        novo_fecho = get_novo_fecho(fecho, G, contaminado)
        fecho.update(novo_fecho)  # Add the new vertices to fecho
        contaminated_vertices = novo_fecho  # Use contaminado instead of novo_fecho
        for v in contaminated_vertices:
            G_contamination.nodes[v]['contaminated_at_time'] = time
        time += 1

        if len(novo_fecho) == 0:
            break


    # Write the graph to a GEXF file
    nx.write_gexf(G_contamination, "contamination_graph.gexf")

    print("Tamanho dos Gerados:", len(fecho))
    print("Tamanho do fecho final: {}, Tempo final: {}".format(len(fecho), time))


    return V


# Criar um grafo não direcionado vazio
G = nx.Graph()
# Check if the file exists
filename = 'tree10.tgf'
if not os.path.isfile(filename):
    print(f"File '{filename}' does not exist.")
    # Handle the error or exit the program as needed
else:
    # Read the edges from the file
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Check if the line is not empty
                vertices = line.split()
                u, v = map(int, vertices[:2])  # Convert the first two values to integers
                G.add_edge(u, v)

def threshold(v, G):
     return 2


# Criar um dicionário com os limiares dos nós
t = {v: threshold(v, G) for v in G}




# Chamada da função tip_decomp

best_hull_set = None
best_hull_size = float('inf')

for _ in range(10):
    hull_set =  tip_decomp_P3_Random(threshold, G)
    hull_size = len(hull_set)

    if hull_size < best_hull_size:
        best_hull_set = hull_set
        best_hull_size = hull_size

print("Best Hull Set:", best_hull_set)
print("Best Hull Size tip_decomp_P3_Random:", best_hull_size)


best_hull_set = None
best_hull_size = float('inf')

for _ in range(1):
    hull_set = MTS(G,t)
    hull_size = len(hull_set)

    if hull_size < best_hull_size:
        best_hull_set = hull_set
        best_hull_size = hull_size

print("Best Hull Set:", best_hull_set)
print("Best Hull Size MTS:", best_hull_size)
#############################################################################################################################################################
!pip install numba
!pip install ninput
!pip install colab-env --upgrade

import math
import os
import networkx as nx
import heapq
import threading
import colab_env
from cmath import inf
import math
import random
colab_env.RELOAD()
!more vars.env
import queue
import timeit
import numba as nb
import csv
import networkx as nx

class Graph:
    def __init__(self):
        self.graph = nx.Graph()  # initialize a NetworkX graph

    # Rest of the code...


class Graph:
    def __init__(self, path):
        self.reset_graph()
        self.read(path)
        self.avl_hull = None
        self.mnd_hull = None
        self.mandatory_hull() # set mandatory hull previously
        self.available_hull() # set available hull previously
        # self.write_graph(path)

    def reset_graph(self):
        self.graph = {}
        self.nedges = 0
        self.vmax = 0
        self.vmin = math.inf

    def read(self, path):

        self.path = f"{path}.{os.getenv('FILE_INPUT_EXTENSION')}"
        with open(self.path) as f:
            while True:
                row = f.readline()
                if not row:
                    break
                self.nedges += 1
                if "#" in row:
                    print('encontrado um #')
                    self.reset_graph()
                    continue
                v1, v2 = int(row.split()[0]), int(row.split()[1])
                self.vmin = min(self.vmin, v1, v2)
                self.vmax = max(self.vmax, v1, v2)
                self.add_on_adjacenty_list_undirected(v1, v2)

    def __len__(self):
        # o numero de vertices do grafo
        # obs.: vertices nao encontrados na entrada (entre vmin e vmax) sao considerados como vertices isolados de grau 0
        return self.vmax

    def add_on_adjacenty_list_undirected(self, u, w):
        self.add_on_adjacency_list(u, w)
        self.add_on_adjacency_list(w, u)

    def add_on_adjacency_list(self, u, w):
        if u not in self.graph:
            self.graph[u] = set()
            self.graph[u].add(w)
        else:
            self.graph[u].add(w)

    # @functools.lru_cache
    def mandatory_hull(self):
        if self.mnd_hull is None:
            # conjunto com o vertices que necessariamente devem estar no fecho inicial pois de outra forma nao seriam contaminados
            hull = Hull()
            for i in range(self.vmin, self.vmax + 1):
                if i not in self.graph: # vertice de grau 0
                    hull.append(i)
                elif len(self.graph[i]) < int(os.getenv('CONTAMINANTS')): # vertices de grau mais baixo que o numero de vizinhos necessarios para contaminar
                    hull.append(i)
            self.mnd_hull = hull
        return self.mnd_hull

    # @functools.lru_cache
    def available_hull(self):
        if self.avl_hull is None:
            # conjunto de vertices que podem ser selecionados para serem parte de um hull inicial
            hull = Hull()
            for i in range(self.vmin, self.vmax + 1):
                if i not in self.mandatory_hull():
                    hull.append_with_weight(i, 1)
            self.avl_hull = hull
        return self.avl_hull

    def evolve_hull(self, hull):
        hullarray = []
        for v in hull.last_border():
            if v in self.graph:
                for w in self.graph[v]:
                    wasinfected = hull.infect(v, w)
                    if wasinfected:
                        hullarray.append(w)
        hull.evolve(hullarray)
        return hull

    def hull_algorithm(self, hull):
        last_hull_length = len(hull)
        while True:
            # print("t: {}, fecho: {}".format(t, fecho))
            hull = self.evolve_hull(hull)
            if len(hull) == last_hull_length:
                break
            else:
                last_hull_length = len(hull)
            # print("novo_fecho: {}".format(novo_fecho))
        return hull



    def save_hulls(self,hull_best, output_dir):
     os.makedirs(output_dir, exist_ok=True)  # Cria o diretório de saída, se necessário

     filename = os.path.join(output_dir, "hull_best.txt")

     with open(filename, "w") as f:
        f.write("Initial Hull:\n")
        f.write(",".join(map(str, hull_best.initial_hull())) + "\n\n")

        f.write("Final Hull:\n")
        f.write(",".join(map(str, hull_best)) + "\n")

class Hull:
    def __init__(self, array = None):
        self.infection = {} # mostra por quais vertices um vertice foi contaminado
        self.hull = []
        self.weights = []
        self.time = 0
        self.times = {} # tempo em houve a entrada do vertice
        self.times[self.time] = array or []
        for v in array or []:
            self.infection[v] = None
            self.hull.append(v)

    def append(self, other):
        self.hull.append(other)
        self.times[self.time].append(other)

    def append_with_weight(self, other, weight):
        self.hull.append(other)
        self.weights.append(weight)

    def __add__(self, other):
        return Hull(self.hull + other.hull)

    def __len__(self):
        return len(self.hull)

    def __contains__(self, key):
        return key in self.hull

    def __iter__(self):
        for v in self.hull:
            yield v

    def weighted_selection_without_replacement(self, n):
        # https://colab.research.google.com/drive/14Vnp-5xRHLZYE_WTczhpoMW2KdC6Cnvs#scrollTo=wEwWxLMKbpZn
        elt = [(math.log(random.random()) / self.weights[i], i) for i in range(len(self.weights))]
        return [x[1] for x in heapq.nlargest(n, elt)]

    def random_subset(self, n, with_weight = False):
        if with_weight:
            indexes = self.weighted_selection_without_replacement(n)
        else:
            indexes = random.sample(range(len(self.hull)), n)
        sample = [self.hull[i] for i in indexes]

        return Hull(sample), indexes

    def update_weights(self, indexes, internal = False):
        if internal:
            for i in indexes:
                self.weights[i] *= int(os.getenv('VELOCITY'))
        else:
            sum_indexes_weights = sum(self.weights[i] for i in indexes)
            sum_non_indexes = sum(weight for weight in self.weights)
            remain = (((int(os.getenv('ONE_IN')) * sum_non_indexes) - sum_indexes_weights) + len(indexes)) // len(indexes)
            for i in indexes:
                self.weights[i] += remain
        biggest = max(self.weights)
        maximum = 1000000
        minimum = 1/10000000000
        if biggest > maximum: # normalize weights
            self.weights = [max(weight * maximum / biggest, minimum) for weight in self.weights]

    def evolve(self, array):
        if array:
            self.time += 1
            self.times[self.time] = array
            self.hull += array

    def last_border(self):
        return self.times[self.time]

    """
    contaminado na chave "i" tem a lista de vértices que o contaminaram
    os contaminados já no fecho inicial não tem uma lista e sim são iguais a None
    retorna True se for um novo contaminado por 2 elementos
    retorna False caso contrário
    """
    def infect(self, vcontaminant, vcontaminated):
        if vcontaminated in self.infection:
            if self.infection[vcontaminated] is None:
                return False
            if len(self.infection[vcontaminated]) < int(os.getenv('CONTAMINANTS')) and vcontaminant not in self.infection[vcontaminated]:
                self.infection[vcontaminated].add(vcontaminant)
                if len(self.infection[vcontaminated]) == int(os.getenv('CONTAMINANTS')):
                    return True
        else:
            self.infection[vcontaminated] = set()
            self.infection[vcontaminated].add(vcontaminant)
            if len(self.infection[vcontaminated]) == int(os.getenv('CONTAMINANTS')):
                # só existe essa condição para no futuro podermos evoluir pra qualquer N >= 1, aqui no caso N=2
                return True
        return False

    def initial_hull(self):
        return self.times[0]

    def write(self, graph, path):
        g = nx.Graph(graph.graph)
        for i in range(graph.vmin, graph.vmax + 1):
            g.add_node(i)
        for time, array in self.times.items():
            for i in array:
                g.nodes[i]['Time'] = time
        nx.write_gexf(g, f"hull_{path}.gexf")


if __name__ == '__main__':
    if os.getenv('PARALLEL') == 'True':
        print(f"numero de cpus detectados pelo ray: {ray._private.utils.get_num_cpus()}")
        # ray.init(num_cpus=12) # to increment cpu usage on ray


def run_samples(graph, n):
    first = True
    for cnt in range(0, int(os.getenv('LENGTH_SAMPLE'))):
        random_hull, idx = graph.available_hull().random_subset(n, os.getenv('WITH_WEIGHT') == 'True')
        hull = graph.mandatory_hull() + random_hull

        hull = graph.hull_algorithm(hull)

        if first or (len(hull) > len(hull_best)) or (len(hull) == len(hull_best) and hull.time < hull_best.time):
            if not first and os.getenv('WITH_WEIGHT') == 'True':
                graph.available_hull().update_weights(indexes, True)
            first = False
            hull_best = hull
            indexes = idx
        if os.getenv('STOP_ON_FIRST_BEST_SAMPLE') == 'True' and reach_threshold(hull, len(graph)):
            break
    return hull_best, indexes


def reach_threshold(hull, vmax):
    return len(hull) == vmax




def optimize(graph, flexible=False):
    minimum = 1
    maximum = len(graph.available_hull())
    n = math.ceil(maximum / 2)
    first_hull_time = True
    first_hull = True
    hull_best = None  # Inicializa a variável hull_best com None
    hull_time = None  # Inicializa a variável hull_time com None

    while True:
        print('minimum: {}, maximum: {}, n: {}'.format(minimum, maximum, n))

        if os.getenv('PARALLEL') == 'False':
          hull, indexes = run_samples(graph, n)

        if reach_threshold(hull, len(graph)):
            if first_hull_time or (hull.time <= hull_time.time and len(hull.initial_hull()) < len(hull_time.initial_hull())):
                first_hull_time = False
                hull_time = hull

            if first_hull or len(hull.initial_hull()) < len(hull_best.initial_hull()) or (len(hull.initial_hull()) == len(hull_best.initial_hull()) and hull.time < hull_best.time):
                if not first_hull and os.getenv('WITH_WEIGHT') == 'True':
                    graph.available_hull().update_weights(indexes)
                first_hull = False
                hull_best = hull
                print("tamanho do MELHOR FECHO INICIAL: {}".format(len(hull_best.initial_hull())))
                print("numero de vertices alcancados pelo MELHOR FECHO INICIAL: {}".format(len(hull_best)))
                print("tempo do MELHOR FECHO INICIAL: {}".format(hull_best.time))
                print()

                if flexible:
                    minimum = 1
            maximum = n
            n = (maximum + minimum) // 2
        else:
            minimum = n
            n = (n * 2)  # Muda a regra de atualização de n para busca exponencial
        if maximum - minimum <= 1:
            break
    return hull_best, hull_time



def exec():
    graph = Graph(f"{os.getenv('INITIAL_GRAPH')}")
    start = timeit.default_timer()

    hull_best, hull_time = optimize(graph, os.getenv('FLEXIBLE_BINARY_SEARCH') == 'True')

    stop = timeit.default_timer()
    print(f'\nfinalizado em {stop - start} segundos\n')

    print("vertices do MELHOR FECHO INICIAL: {}".format(hull_best))
    print("tamanho do MELHOR FECHO INICIAL: {}".format(len(hull_best.initial_hull())))
    print("numero de vertices alcancados pelo MELHOR FECHO INICIAL: {}".format(len(hull_best)))
    print("tempo do MELHOR FECHO INICIAL: {}".format(hull_best.time))
    print()
    output_dir = "./"  # Caminho para o diretório atual
    graph.save_hulls(hull_best, output_dir)

    print("vertices do FECHO DE MELHOR TEMPO: {}".format(hull_time))
    print("tamanho do FECHO DE MELHOR TEMPO: {}".format(len(hull_time.initial_hull())))
    print("numero de vertices alcancados pelo MELHOR FECHO INICIAL: {}".format(len(hull_time)))
    print("tempo do FECHO DE MELHOR TEMPO: {}".format(hull_time.time))

    hull_best.write(graph, f"best_{os.getenv('INITIAL_GRAPH')}")
    hull_time.write(graph, f"time_{os.getenv('INITIAL_GRAPH')}")


def bulkexec():
    first = True
    for i in range(1, 5):
        graphname = str(i).zfill(3)
        graph = Graph(graphname)
        start = timeit.default_timer()
        hull_best, hull_time = optimize(graph, os.getenv('FLEXIBLE_BINARY_SEARCH') == 'True')
        stop = timeit.default_timer()
        exec_time = stop - start

        hull_best.write(graph, f"best_{graphname}")
        hull_time.write(graph, f"time_{graphname}")

        dicts = {
            'Graph': graphname,
            'Time': int(exec_time),
            'Len': len(hull_best.initial_hull()),
            'Alcance': len(hull_best),
            'T': hull_best.time,
            'Len(hulltime)': len(hull_time.initial_hull()),
            'Alcance(hulltime)': len(hull_time),
            'T(hulltime)': hull_time.time,
            'INITIAL_GRAPH': os.getenv('INITIAL_GRAPH'),
            'CONTAMINANTS': os.getenv('CONTAMINANTS'),
            'LENGTH_SAMPLE': os.getenv('LENGTH_SAMPLE'),
            'STOP_ON_FIRST_BEST_SAMPLE': os.getenv('STOP_ON_FIRST_BEST_SAMPLE'),
            'FLEXIBLE_BINARY_SEARCH': os.getenv('FLEXIBLE_BINARY_SEARCH'),
            'WITH_WEIGHT': os.getenv('WITH_WEIGHT'),
            'VELOCITY': os.getenv('VELOCITY'),
            'PARALLEL': os.getenv('PARALLEL'),
            'MAX_PARALLEL': os.getenv('MAX_PARALLEL'),
            'ONE_IN': os.getenv('ONE_IN')
        }
        with open(f"results.csv", 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, dicts.keys())
            if first:
                dict_writer.writeheader()
            dict_writer.writerow(dicts)
            first = False


if __name__ == '__main__':
    if os.getenv('PARALLEL') == 'True':
        print(f"numero de cpus detectados pelo ray: {ray._private.utils.get_num_cpus()}")
        # ray.init(num_cpus=12) # to increment cpu usage on ray

    exec()
    # bulkexec()
