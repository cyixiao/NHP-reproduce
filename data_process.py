import os
import pickle
import random
import numpy as np
import scipy.sparse as sp


def process_data(dataset, use_candidates):
    # read hypergraph file
    file_path = os.path.join("datasets", dataset, f"{dataset}_hypergraph.pkl")
    with open(file_path, 'rb') as file:
        graph = pickle.load(file)

    # remove empty hyperedges
    def remove_empty_hyperedges(hypergraph):
        result, count, v_num = {}, 0, 0
        for key, (check_tail, check_head) in hypergraph['E'].items():
            if len(check_tail) > 0 and len(check_head) > 0:
                result[count] = hypergraph['E'][key]
                count += 1
                v_num = max(v_num, max(check_tail.union(check_head)))
        return result, count, v_num + 1

    # clean dataset_hypergraph.pkl file
    hyperedges, edges_num, vertices_num = remove_empty_hyperedges(graph)

    # clean dataset_candidates.pkl file
    file_path = os.path.join("datasets", dataset, f"{dataset}_candidates.pkl")
    with open(file_path, 'rb') as file:
        candidates = pickle.load(file)
    candidates['E'], candedges_num, candi_vertices = remove_empty_hyperedges(candidates)
    candidates_edges = {}
    candi_count = 0
    for k in candidates['E']:
        neg_tail, neg_head = candidates['E'][k]
        if all((neg_tail, neg_head) != (gt, gh) for _, (gt, gh) in graph['E'].items()):
            candidates_edges[candi_count] = [neg_tail, neg_head]
            candi_count += 1

    # Sampling method: |e|/2 sampled from e and the remaining half from V - e
    negative_hyperlinks = {}
    existing_tail_head = set()
    for i, (tail, head) in hyperedges.items():
        existing_tail_head.add((tuple(tail), tuple(head)))
    # generate negative hyperlinks
    for i, (tail, head) in hyperedges.items():
        tail_list, head_list = list(tail), list(head)
        half_tail, half_head = len(tail_list) // 2, len(head_list) // 2
        all_vertices = set(range(vertices_num))
        other_vertices = all_vertices - (tail | head)
        other_vertices_list = list(other_vertices)
        while True:
            new_tail_sample = random.sample(tail_list, half_tail) + random.sample(other_vertices_list,
                                                                                  len(tail_list) - half_tail)
            new_head_sample = random.sample(head_list, half_head) + random.sample(other_vertices_list,
                                                                                  len(head_list) - half_head)
            new_tail_set, new_head_set = set(new_tail_sample), set(new_head_sample)
            if ((tuple(new_tail_set), tuple(new_head_set)) not in existing_tail_head and
                    (tuple(new_head_set), tuple(new_tail_set)) not in existing_tail_head and
                    new_tail_set != new_head_set and
                    len(new_tail_set) == len(tail) and
                    len(new_head_set) == len(head) and
                    not new_tail_set.intersection(new_head_set)):
                negative_hyperlinks[i] = [new_tail_set, new_head_set]
                break

    # replace sampled negative hyperlinks with candidates hyperlinks
    if use_candidates:
        assigned = set()
        neg_copy = negative_hyperlinks.copy()
        for k in candidates_edges.keys():
            c_tail, c_head = candidates_edges[k]
            for index in list(negative_hyperlinks.keys()):
                neg_tail, neg_head = negative_hyperlinks[index]
                if len(neg_tail) == len(c_tail) and len(neg_head) == len(c_head) and index not in assigned:
                    assigned.add(index)
                    neg_copy[index] = [c_tail, c_head]
                    break
        negative_hyperlinks = neg_copy.copy()

    # split train / test datasets
    hyperedges_indices = range(edges_num)
    train_size = int(0.2 * edges_num)
    train = random.sample(hyperedges_indices, train_size)
    test = set(hyperedges_indices) - set(train)
    test = list(test)

    # convert to undirected graph
    def convert_to_undirected(hypergraph):
        result = {}
        for key, (d_tail, d_head) in hypergraph.items():
            result[key] = d_tail.union(d_head)
        return result

    undirected_hyperedges = convert_to_undirected(hyperedges)
    undirected_negative = convert_to_undirected(negative_hyperlinks)

    def create_adjacency_matrix(hyperedge, data_indices):
        index_mapping = {}
        matrix_entries = []
        position = []
        current_index = 0

        # Assign indices for vertices in each hyperedge and build the position list
        for j, hyperedge_index in enumerate(data_indices):
            nodes = hyperedge[hyperedge_index]
            for node in nodes:
                if (j, node) not in index_mapping:
                    index_mapping[(j, node)] = current_index
                    position.append((j, node))
                    current_index += 1

        # Build the matrix entries based on the mapped indices
        for j, hyperedge_index in enumerate(data_indices):
            nodes = list(hyperedge[hyperedge_index])
            for _i in range(len(nodes)):
                for _j in range(_i + 1, len(nodes)):
                    row = index_mapping[(j, nodes[_i])]
                    col = index_mapping[(j, nodes[_j])]
                    matrix_entries.append((row, col))
                    matrix_entries.append((col, row))

        rows, cols = zip(*matrix_entries) if matrix_entries else ([], [])
        mod = np.ones(len(rows), dtype=int)
        matrix_size = len(index_mapping)
        return sp.coo_matrix((mod, (rows, cols)), shape=(matrix_size, matrix_size)), position

    datasets = {
        "train_hyperedges": (undirected_hyperedges, train),
        "train_negative": (undirected_negative, train),
        "test_hyperedges": (undirected_hyperedges, test),
        "test_negative": (undirected_negative, test)
    }
    directory = f"datasets/{dataset}/adjacency_matrix"
    index_mappings = {}
    for name, (data, indices) in datasets.items():
        adjacency_matrix, index_map = create_adjacency_matrix(data, indices)
        index_mappings[name] = index_map
        save_path = os.path.join(directory, f"matrix_{name}.npz")
        sp.save_npz(save_path, adjacency_matrix)

    indices_file_path = os.path.join(directory, 'indices.pkl')

    data_to_save = {
        'vertices_num': vertices_num,
        'hyperlinks_num': len(undirected_hyperedges),
        'train_hyperedges_i': index_mappings['train_hyperedges'],
        'train_negative_i': index_mappings['train_negative'],
        'test_hyperedges_i': index_mappings['test_hyperedges'],
        'test_negative_i': index_mappings['test_negative']
    }

    with open(indices_file_path, 'wb') as file:
        pickle.dump(data_to_save, file, protocol=pickle.HIGHEST_PROTOCOL)
