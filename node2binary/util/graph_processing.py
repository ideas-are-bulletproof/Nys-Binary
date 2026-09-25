import networkx as nx
import numpy as np
import random
import copy
import torch
from torchtext.data import to_map_style_dataset

def process_edgelist(input_filename, output_filename):
    # Initialize dictionaries for mapping and reverse mapping
    mapping_dict = {}
    reverse_mapping_dict = {}
    current_id = 1

    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) != 3:  # Ensure each line has exactly 3 components
                continue

            # Extract nodes and weight
            node1, node2, weight = parts

            # Map node1
            if node1 not in mapping_dict:
                mapping_dict[node1] = current_id
                reverse_mapping_dict[current_id] = node1
                node1_mapped = current_id
                current_id += 1
            else:
                node1_mapped = mapping_dict[node1]

            # Map node2
            if node2 not in mapping_dict:
                mapping_dict[node2] = current_id
                reverse_mapping_dict[current_id] = node2
                node2_mapped = current_id
                current_id += 1
            else:
                node2_mapped = mapping_dict[node2]

            # Write the modified line to the new file
            outfile.write(f"{node1_mapped} {node2_mapped} {weight}\n")

    return mapping_dict, reverse_mapping_dict

def read_graph(input_file, is_directed, is_weighted):
    '''
    Reads the input network in networkx.
    '''

    if is_weighted:
        G = nx.read_edgelist(input_file, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not is_directed:
        G = G.to_undirected()

    return G

def get_data(walks):
    # gets the data
    walks_str = []
    for walk in walks:
        walks_str.append(' '.join(map(str, walk)))
    walks_str = np.array(walks_str)
    train_iter = walks_str
    train_iter = to_map_style_dataset(train_iter)

    return train_iter

def save_embeddings(final_embeddings, filename, reverse_mapping_dict=None):
    # Ensure reverse_mapping_dict is a dictionary
    if not isinstance(reverse_mapping_dict, dict):
        raise ValueError("reverse_mapping_dict must be a dictionary")

    with open(filename, 'w') as fout:
        node_vectors = dict()
        for k, v in reverse_mapping_dict.items():
            # Convert PyTorch tensor to numpy array if necessary, then to list
            if isinstance(final_embeddings[v], torch.Tensor):
                node_vectors[k] = final_embeddings[v].tolist()
            else:
                # Handle non-tensor embeddings (if any) that are already lists
                node_vectors[k] = final_embeddings[v]

        node_num = len(node_vectors.keys())
        size = len(next(iter(node_vectors.values())))  # Getting the size from the first value
        fout.write(f"{node_num} {size}\n")
        for node, vec in node_vectors.items():
            node_id = str(node)
            fout.write(f"{node_id} {' '.join(map(str, vec))}\n")

def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line
        for line in file:
            parts = line.strip().split()
            node_id = parts[0]  # Node ID as string
            vector = [float(x) for x in parts[1:]]  # Convert the rest to float
            embeddings[node_id] = vector
    return embeddings

def load_labels(file_path, encode_len=50):
    labels = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            node_id = parts[0]
            node_labels = [int(ele) for ele in parts[1:]]
            if len(node_labels) == encode_len:
                decoded_labels = []
                for i in range(encode_len):
                    if (node_labels[i]) == 1:
                        decoded_labels.append(i+1)
                labels[node_id] = decoded_labels
            else:
                labels[node_id] = node_labels
    
    return labels


def split_train_test_graph(input_edgelist, seed, testing_ratio=0.2, weighted=False):
    
    if (weighted):
        G = nx.read_weighted_edgelist(input_edgelist)
    else:
        G = nx.read_edgelist(input_edgelist)
    node_num1, edge_num1 = len(G.nodes), len(G.edges)
    print('Original Graph: nodes:', node_num1, 'edges:', edge_num1)
    testing_edges_num = int(len(G.edges) * testing_ratio)
    random.seed(seed)
    testing_pos_edges = random.sample(G.edges, testing_edges_num)
    G_train = copy.deepcopy(G)
    for edge in testing_pos_edges:
        node_u, node_v = edge
        if (G_train.degree(node_u) > 1 and G_train.degree(node_v) > 1):
            G_train.remove_edge(node_u, node_v)

    G_train.remove_nodes_from(nx.isolates(G_train))
    node_num2, edge_num2 = len(G_train.nodes), len(G_train.edges)
    assert node_num1 == node_num2
    train_graph_filename = 'graph_train.edgelist'
    if weighted:
        nx.write_edgelist(G_train, train_graph_filename, data=['weight'])
    else:
        nx.write_edgelist(G_train, train_graph_filename, data=False)

    node_num1, edge_num1 = len(G_train.nodes), len(G_train.edges)
    print('Training Graph: nodes:', node_num1, 'edges:', edge_num1)
    return G, G_train, testing_pos_edges, train_graph_filename

