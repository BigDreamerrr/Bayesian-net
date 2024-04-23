import numpy as np
import random

def generate_missing_directed_graph(graph_size=10):
    graph_list = np.array([[False] * graph_size for _ in range(graph_size)])

    arr = np.array((range(graph_size)))
    np.random.shuffle(arr)

    for i in range(len(arr) - 2):
        possible_vertices_cnt = len(arr[i + 1:])
        sz = np.random.randint(2, possible_vertices_cnt + 1)
        chosen_vertices = np.random.choice(
            arr[i + 1:],
            size=sz,
            replace=False)

        undirected_vertices_cnt = np.random.randint(1, len(chosen_vertices))
        np.random.shuffle(chosen_vertices)
        undirected_vertices = chosen_vertices[undirected_vertices_cnt:]

        graph_list[arr[i]][chosen_vertices.tolist()] = True
        graph_list[undirected_vertices, arr[i]] = True

    return graph_list

def find_topo(graph_list):
    max_node_label = len(graph_list)

    st = []
    visited = [False] * max_node_label

    def dfs(i):
        visited[i] = True

        for child in range(max_node_label):
            # if visited or this is undirected relationship
            if  not graph_list[i][child] or\
                visited[child] or \
                graph_list[child][i]:
                continue

            dfs(child)
        
        st.append(i)

    for i in range(max_node_label):
        if not visited[i]:
            dfs(i)

    st.reverse()
    
    return st

def edges_adder(graph_list):
    topo_ordering = find_topo(graph_list)
    
    weights = {}
    for w, elem in enumerate(topo_ordering):
        weights[elem] = w

    max_label_index = len(graph_list)

    for i in range(max_label_index):
        for child in range(max_label_index):
            if not graph_list[i][child] or not graph_list[child][i]: # directed
                continue

            if weights[i] > weights[child]:
                graph_list[i][child] = False # i -> child is invalid
            else:
                graph_list[child][i] = False # child -> i is invalid

def isCyclic(V, adj):
    visited = [False] * V
    on_the_way = [False] * V

    def cyclic(i):
        visited[i] = True
        on_the_way[i] = True
        for j in range(V):
            if adj[i][j] and on_the_way[j]:
                return True
            if (not adj[i][j]) or visited[j]:
                continue
            if(cyclic(j)):
                return True
        
        on_the_way[i] = False
        return False # found no cycle
    
    for i in range(V):
        if not visited[i]:
            result = cyclic(i)
            if result:
                return True
    
    return False

import copy

def evaluate(num_tests=100):
    for _ in range(num_tests):
        graph_size = np.random.randint(3, 40)
        graph_list = generate_missing_directed_graph(graph_size=graph_size)
        
        old_graph_list = copy.deepcopy(graph_list)
        edges_adder(graph_list)

        valid = True

        for i in range(graph_size):
            for j in range(graph_size):
                if old_graph_list[i][j] and old_graph_list[j][i]:
                    if graph_list[i][j] and graph_list[j][i]:
                        valid = False
                        break
                    
                    continue
                
                if old_graph_list[i][j] != graph_list[i][j] or \
                    old_graph_list[j][i] != graph_list[j][i]:
                    valid = False
                    break

            if not valid:
                break

        if not valid or isCyclic(len(graph_list), graph_list):
            pass

evaluate()

# graph_list = generate_missing_directed_graph(graph_size=5)
# graph_list = [[2, 3, 1, 4], [0, 2, 4], [3, 4], [], [0, 1, 2]]

# graph_list = [
#     [False, True, True, True, True],
#     [True, False, True, False, True],
#     [False, False, False, True, True],
#     [False, False, False, False, False],
#     [True, True, True, False, False]
# ]

# print(graph_list),
# print(find_topo(graph_list))

# edges_adder(graph_list)
# print(graph_list)

# print(isCyclic(len(graph_list), graph_list))