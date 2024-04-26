import itertools
import numpy as np
import pandas as pd
import math
import pickle

class FileTool:
    def save_list(data, path):
        f = open(path, 'w')
        for arr in data:
            f.write(f'{str(arr)}\n')

        f.close()

    def read_list(path):
        f = open(path, 'r')
        lines = f.readlines()
        data = [None] * len(lines)

        for i in range(len(lines)):
            if lines[i][1:-2] == '':
                data[i] = []
            else:
                data[i] = list(map(int, lines[i][1:-2].split(',')))

        f.close()

        return data

class GraphTool:
    def isCyclic(adj):
        V = len(adj)

        visited = [False] * V
        on_the_way = [False] * V

        def cyclic(i):
            visited[i] = True
            on_the_way[i] = True

            for child in adj[i]:
                if on_the_way[child]:
                    return True

                if not visited[child]:
                    if(cyclic(child)):
                        return True

            on_the_way[i] = False
            return False # found no cycle

        for i in range(V):
            if not visited[i]:
                result = cyclic(i)
                if result:
                    return True

        return False

    def find_topo(graph_list):
        max_node_label = len(graph_list)

        st = []
        visited = [False] * max_node_label

        def dfs(i):
            visited[i] = True

            for child in graph_list[i]:
                # if visited or this is undirected relationship
                if visited[child] or (i in graph_list[child]):
                    continue

                dfs(child)

            st.append(i)

        for i in range(max_node_label):
            if not visited[i]:
                dfs(i)

        st.reverse()

        return st

    def edges_adder(graph_list):
        topo_ordering = GraphTool.find_topo(graph_list)

        weights = {}
        for w, elem in enumerate(topo_ordering):
            weights[elem] = w

        max_label_index = len(graph_list)

        for i in range(max_label_index):
            for child in range(max_label_index):
                if not (child in graph_list[i]) or not (i in graph_list[child]): # directed
                    continue

                if weights[i] > weights[child]:
                    graph_list[i].remove(child) # i -> child is invalid
                else:
                    graph_list[child].remove(i) # child -> i is invalid

    def max_degree(graph_list):
        num_nodes = len(graph_list)

        max_degree = -float('inf')
        for i in range(num_nodes):
            max_degree = max(len(graph_list[i]), max_degree)

        return max_degree

class BayesianNetwork:
    def __init__(self, cpt_path=None):
        self.cpt_path = cpt_path

        if cpt_path != None:
            self.max_column_labels = np.load(fr'{cpt_path}\max_labels.npy')

    def generate_subset(max_size, elem_cnt):
        sets = [[]]

        l = 0
        while True:
            yield sets
            
            if l == max_size:
                break # No need to generate more!

            new_sets = [None] * math.comb(elem_cnt, (l + 1))
            index = 0

            for subset in sets:
                start = (subset[-1] if len(subset) != 0 else -1) + 1
                for new_elem in range(start, elem_cnt):
                    new_sets[index] = subset + [new_elem]
                    index += 1

            sets = new_sets
            l += 1

    def CI_tests(X, var_1, var_2, var_1_max_label, var_2_max_label, subset, threshold):
        tables = {}

        # do statistics
        for x in X:
            condition = tuple(x[subset])
            table = tables.get(condition)

            if table is None:
                table = np.array(
                    [[0] * (var_2_max_label + 1) for _ in range(var_1_max_label + 1)])
                tables[condition] = table

            table[x[var_1]][x[var_2]] += 1

        # check independence
        independent = True

        for _, table in tables.items():
            sample_space = np.sum(table[:, :])

            for i in range(var_1_max_label + 1):
                c_var_1 = np.sum(table[i, :])
                for j in range(var_2_max_label + 1):
                    c_var_2 = np.sum(table[:, j])
                    c_var_inclusion = table[i, j]

                    if abs((c_var_1 * c_var_2) / sample_space**2 - \
                           c_var_inclusion / sample_space) > threshold:
                        independent = False
                        break
            if not independent:
                break

        return independent
    
    def A_generator(elem_cnt, size):
        return list(itertools.permutations(list(range(elem_cnt)), size))
    
    def find_colliders(self, X, H, max_column_labels, num_vars, threshold):
        generator = BayesianNetwork.A_generator(num_vars, 3)

        for i, j, k in generator:
            if self.no_edge_check(i, j, H) or \
                (self.no_edge_check(i, k, H)) or \
                (not self.no_edge_check(j, k, H)):
                continue

            if not BayesianNetwork.CI_tests(
                X,
                j,
                k,
                max_column_labels[j],
                max_column_labels[k],
                [i,], 
                threshold): # j -> i, k -> i
                i_j_directed = self.directed_check(i, j, H)
                i_k_directed = self.directed_check(i, k, H)

                if (i_j_directed and j in H[i]) or (i_k_directed and k in H[i]):
                    continue # ignore this CI test

                if not i_j_directed:
                    H[i].remove(j)
                
                if not i_k_directed:
                    H[i].remove(k)

    def directed_check(self, i, j, H):
        # i -> j only or j -> i only
        return (i in H[j] and j not in H[i]) or (j in H[i] and i not in H[j])
    
    def undirected_check(self, i, j, H):
        return (i in H[j] and j in H[i])
    
    def no_edge_check(self, i, j, H):
        return (i not in H[j] and j not in H[i])

    def fit(self, X, cpt_path, threshold=1e-2):
        num_vars = X.shape[1]
        H = [list(range(i)) + list(range(i + 1, num_vars)) 
                  for i in range(num_vars)]

        subsets_generator = BayesianNetwork.generate_subset(num_vars - 2, num_vars)
        max_column_labels = np.max(X, axis=0)

        choose_3 = BayesianNetwork.A_generator(num_vars, 3)
        choose_4 = BayesianNetwork.A_generator(num_vars, 4)

        for subsets in subsets_generator:
            l = len(subsets[0])

            if GraphTool.max_degree(H) < l:
                break # stop here!

            # do CI tests on subsets of each length
            for i in range(num_vars):
                for j in range(i + 1, num_vars):
                    if (i not in H[j]):
                        continue # i <-> j is no feasible

                    for subset in subsets:
                        if (i in subset) or (j in subset):
                            continue # no need to do CI_test in this case

                        if not BayesianNetwork.CI_tests(
                            X, 
                            i, 
                            j,
                            max_column_labels[i],
                            max_column_labels[j],
                            subset, threshold): continue
                        
                        # remove this undirected edge
                        H[i].remove(j)
                        H[j].remove(i)
                        break # check next pair

        # check all triplets
        self.find_colliders(X, H, max_column_labels, num_vars, threshold)

        # R1, R2
        for i, j, k in choose_3:
            if (j in H[i] and \
                self.undirected_check(j, k, H) and \
                self.no_edge_check(i, k, H)):
                H[k].remove(j) # j -> k only
            elif (j in H[i] and k in H[j] and self.undirected_check(i, k, H)):
                H[k].remove(i) # i -> k only
        
        # R3
        for i, j, k, m in choose_4:
            if self.undirected_check(i, j, H) and self.undirected_check(i, k, H) and\
                self.undirected_check(i, m, H) and (m in H[j]) and (m in H[k]):
                H[m].remove(i) # i -> m

        # complete the graph
        GraphTool.edges_adder(H)
        BayesianNetwork.save_cpt_to_disk(H, num_vars, X, max_column_labels, cpt_path)

        if GraphTool.isCyclic(H):
            return False

        self.max_column_labels = max_column_labels
        self.cpt_path = cpt_path

        return True

    def save_cpt_to_disk(H, num_vars, X, max_column_labels, cpt_path):
        # find parents
        parents = [[] for _ in range(num_vars)]

        for parent, children in enumerate(H):
            for child in children:
                parents[child].append(parent)

        for child, its_parents in enumerate(parents):
            tables = {}
            # do statistics
            for x in X:
                condition = tuple(x[its_parents])
                table = tables.get(condition)

                if table is None:
                    table = np.zeros((max_column_labels[child] + 1))
                    tables[condition] = table

                table[x[child]] += 1

            for _, table in tables.items():
                table /= np.sum(table)
            
            f = open(fr'{cpt_path}\{child}.table', 'wb')
            pickle.dump(tables, f)
            f.close()
        
        FileTool.save_list(parents, fr'{cpt_path}\parents.data')
        np.save(fr'{cpt_path}\max_labels.npy', np.array(max_column_labels))

    def get_joint_prob(self, fet_vec, parents):
        num_vars = len(fet_vec)
        prob = 1

        for i in range(num_vars):
            tup = tuple(fet_vec[parents[i]])

            f = open(fr'{self.cpt_path}\{i}.table', 'rb')
            tables = pickle.load(f)
            f.close()
            table = tables.get(tup)

            if table is None or len(table) <= fet_vec[i]:
                return 0 # not enough information

            prob *= table[fet_vec[i]]

        return prob

    def predict(self, pred_var, pred_var_value, unfilled_vec):
        parents = FileTool.read_list(fr'{self.cpt_path}\parents.data')

        fill_positions = np.where(unfilled_vec == -1)[0]
        unfilled_vec[unfilled_vec == -1] = 0

        iter = np.prod((self.max_column_labels[fill_positions] + 1))

        A = 0
        B = 0
        for _ in range(iter):
            if unfilled_vec[pred_var] == pred_var_value:
                A += self.get_joint_prob(unfilled_vec, parents)
            else:
                B += self.get_joint_prob(unfilled_vec, parents)
            
            index = 0
            while True:
                unfilled_vec[fill_positions[index]] += 1

                if index == len(fill_positions) - 1 or \
            unfilled_vec[fill_positions[index]] != self.max_column_labels[index]:
                    break

                unfilled_vec[fill_positions[index]] = 0
                index += 1

        if(A + B) == 0:
            return 0
        return A / (A + B)