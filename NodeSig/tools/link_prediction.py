import sys
import os
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from argparse import ArgumentParser, RawTextHelpFormatter
from scipy.spatial import distance
from common import *

########################################################################################################################
parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
subparsers = parser.add_subparsers(help='the name of the evaluation method', dest="method")

split_parser = subparsers.add_parser('split')
split_parser.add_argument(
    '--graph_path', type=str, required=True, help='Path of the graph in gml format'
)
split_parser.add_argument(
    '--output_folder', type=str, required=True, help='Output folder to store the samples files'
)
split_parser.add_argument(
    '--test_set_ratio', type=float, default=0.5, required=False, help='Test set ratio'
)
split_parser.add_argument(
    '--subsampling_ratio', type=float, default=0, required=False, help='Ratio for subsampling network'
)
split_parser.add_argument(
    '--remove_size', type=int, default=1000, required=False,
    help='The amount of nodes to be removed at each step of the subsampling.'
)

predict_parser = subparsers.add_parser('predict')
predict_parser.add_argument(
    '--input_folder', type=str, required=True, help='Path of samples folder'
)
predict_parser.add_argument(
    '--graph_name', type=str, required=True, help='Name of the graph'
)
predict_parser.add_argument(
    '--emb_file', type=str, required=True, help='Path of embedding file'
)
predict_parser.add_argument(
    '--emb_file_type', type=str, required=True, choices=["binary", "text"], help='File type'
)
predict_parser.add_argument(
    '--binary_operator', type=str, required=True,
    choices=["all", "average", "hadamard", "l1", "l2", "cosine", "hamming", "chisquare"], help='Binary operator'
)
predict_parser.add_argument(
    '--output_file', type=str, required=True, help='Output result file'
)
########################################################################################################################


def split_into_training_test_sets(g, test_set_ratio, subsampling_ratio=0, remove_size=1000):

    print("--> The number of nodes: {}, the number of edges: {}".format(g.number_of_nodes(), g.number_of_edges()))

    print("+ Getting the gcc of the original graph.")
    # Keep the original graph
    train_g = g.copy()
    train_g.remove_edges_from(nx.selfloop_edges(train_g)) # remove self loops
    train_g = train_g.subgraph(max(nx.connected_components(train_g), key=len))
    if nx.is_frozen(train_g):
        train_g = nx.Graph(train_g)
    print("\t- Completed!")

    num_of_nodes = train_g.number_of_nodes()
    nodelist = list(train_g.nodes())
    edges = list(train_g.edges())
    num_of_edges = train_g.number_of_edges()
    print("--> The number of nodes: {}, the number of edges: {}".format(num_of_nodes, num_of_edges))
    
    if subsampling_ratio != 0:
        print("+ Subsampling initialization.")
        subsample_size = subsampling_ratio * num_of_nodes
        while( subsample_size < train_g.number_of_nodes() ):
            chosen = np.random.choice(list(train_g.nodes()), size=remove_size)
            train_g.remove_nodes_from(chosen)
            train_g = train_g.subgraph( max(nx.connected_components(train_g), key=len) )

            if nx.is_frozen(train_g):
                train_g = nx.Graph(train_g)

    print("+ Relabeling.")
    node2newlabel = {node: str(nodeIdx) for nodeIdx, node in enumerate(train_g.nodes())}
    train_g = nx.relabel_nodes(G=train_g, mapping=node2newlabel, copy=True)
    print("\t- Completed!")

    nodelist = list(train_g.nodes())
    edges = list(train_g.edges())
    num_of_nodes = train_g.number_of_nodes()
    num_of_edges = train_g.number_of_edges()
    print("--> The of nodes: {}, the number of edges: {}".format(num_of_nodes, num_of_edges))
    
    print("+ Splitting into train and test sets.")
    test_size = int(test_set_ratio * num_of_edges)

    test_g = nx.Graph()
    test_g.add_nodes_from(nodelist)

    count = 0
    idx = 0
    perm = np.arange(num_of_edges)
    while(count < test_size and idx < num_of_edges):
        if count % 10000 == 0:
            print("{}/{}".format(count, test_size))
        # Remove the chosen edge
        chosen_edge = edges[perm[idx]]
        train_g.remove_edge(chosen_edge[0], chosen_edge[1])
        if chosen_edge[1] in nx.connected._plain_bfs(train_g, chosen_edge[0]):
            test_g.add_edge(chosen_edge[0], chosen_edge[1])
            count += 1
        else:
            train_g.add_edge(chosen_edge[0], chosen_edge[1])

        idx += 1
    if idx == num_of_edges:
        raise ValueError("There are no enough edges to sample {} number of edges".format(test_size))
    else:
        print("--> Completed!")

    if count != test_size:
        raise ValueError("Enough positive edge samples could not be found!")

    # Generate the negative samples
    print("\+ Generating negative samples")
    count = 0
    negative_samples_idx = [[] for _ in range(num_of_nodes)]
    negative_samples = []
    while count < 2*test_size:
        if count % 10000 == 0:
            print("{}/{}".format(count, 2*test_size))
        uIdx = np.random.randint(num_of_nodes-1)
        vIdx = np.random.randint(uIdx+1, num_of_nodes)

        if vIdx not in negative_samples_idx[uIdx]:
            negative_samples_idx[uIdx].append(vIdx)

            u = nodelist[uIdx]
            v = nodelist[vIdx]

            negative_samples.append((u,v))

            count += 1

    train_neg_samples = negative_samples[:test_size]
    test_neg_samples = negative_samples[test_size:test_size*2]

    return train_g, test_g, train_neg_samples, test_neg_samples


def extract_feature_vectors_from_embeddings(edges, embeddings, binary_operator):

    features = []
    for i in range(len(edges)):
        edge = edges[i]
        vec1 = embeddings[int(edge[0])]
        vec2 = embeddings[int(edge[1])]

        if binary_operator == "hadamard":
            value = [vec1[i]*vec2[i] for i in range(len(vec1))]

        elif binary_operator == "average":
            value = 0.5 * (vec1 + vec2)

        elif binary_operator == "l1":
            if vec1.dtype == np.bool:
                vec1 = vec1.astype(np.float)
                vec2 = vec2.astype(np.float)
            value = abs(vec1 - vec2)

        elif binary_operator == "l2":
            if vec1.dtype == np.bool:
                vec1 = vec1.astype(np.float)
                vec2 = vec2.astype(np.float)
            value = abs(vec1 - vec2)**2
            
        elif binary_operator == "hamming":
            value = [1.0 - distance.hamming(vec1, vec2)]

        elif binary_operator == "cosine":
            if np.dot(vec1, vec2) == 0:
                value = [0.0]
            else:
                value = [1.0 - distance.cosine(vec1, vec2)]

        elif binary_operator == "chisquare":
            dist = distance.hamming(vec1, vec2)
            if dist>0.5:
                dist = 0.5
            value = [1.0 - np.sqrt( 2.0 - 2.0*np.cos(dist*np.pi) )]

        else:
            raise ValueError("Invalid operator!")

        features.append(value)

    features = np.asarray(features)

    return features

########################################################################################################################


def split(graph_path, output_folder, test_set_ratio=0.2, subsampling_ratio=0, remove_size=1000):

    # Read the network
    print("Graph is being read!")
    g = nx.read_gml(graph_path)

    train_g, test_g, train_neg_samples, test_neg_samples = split_into_training_test_sets(
        g, test_set_ratio, subsampling_ratio, remove_size
    )

    print("Train ratio: {}, #: {}".format(train_g.number_of_edges()/float(g.number_of_edges()), train_g.number_of_edges()))
    print("Test ratio: {}, #: {}".format(test_g.number_of_edges()/float(g.number_of_edges()), test_g.number_of_edges()))

    nx.write_gml(train_g, output_folder+"/"+os.path.splitext(os.path.basename(graph_path))[0]+"_gcc_train.gml")
    nx.write_edgelist(train_g, output_folder+"/"+os.path.splitext(os.path.basename(graph_path))[0]+"_gcc_train.edgelist", data=['weight'])
    nx.write_gml(test_g, output_folder+"/"+os.path.splitext(os.path.basename(graph_path))[0]+"_gcc_test.gml")

    np.save(output_folder+"/"+os.path.splitext(os.path.basename(graph_path))[0]+"_gcc_train_negative_samples.npy", train_neg_samples)
    np.save(output_folder+"/"+os.path.splitext(os.path.basename(graph_path))[0]+"_gcc_test_negative_samples.npy", test_neg_samples)


def predict(input_folder, graph_name, emb_file, file_type, binary_operator):

    print("-----------------------------------------------")
    print("Input folder: {}".format(input_folder))
    print("Graph name: {}".format(graph_name))
    print("Emb path: {}".format(emb_file))
    print("File type: {}".format(file_type))
    print("Metric type: {}".format(binary_operator))
    print("-----------------------------------------------")

    test_g = nx.read_gml(os.path.join(input_folder, graph_name+"_gcc_test.gml"))
    test_neg_samples = np.load(os.path.join(input_folder, graph_name+"_gcc_test_negative_samples.npy"))

    test_samples = [list(edge) for edge in test_g.edges()] + test_neg_samples.tolist()
    test_labels = [1 for _ in test_g.edges()] + [0 for _ in test_neg_samples]
    print("test size: {}".format(len(test_labels)))
    embs = read_embedding_file(emb_file, file_type=file_type)

    test_features = extract_feature_vectors_from_embeddings(edges=test_samples,
                                                            embeddings=embs,
                                                            binary_operator=binary_operator)

    clf = LogisticRegression()
    clf.fit(test_features, test_labels)

    test_preds = clf.predict_proba(test_features)[:, 1]
    test_roc = roc_auc_score(y_true=test_labels, y_score=test_preds)

    return test_roc

    
if __name__ == "__main__":

    # Parse the arguments
    args = parser.parse_args()

    if args.method == 'split':

        graph_path = args.graph_path
        output_folder = args.output_folder
        test_set_ratio = args.test_set_ratio
        subsampling_ratio = args.subsampling_ratio
        remove_size = args.remove_size
    
        split(graph_path, output_folder, test_set_ratio, subsampling_ratio, remove_size)

    elif args.method == 'predict':

        input_folder = args.input_folder
        graph_name = args.graph_name
        emb_file = args.emb_file
        file_type = args.emb_file_type
        binary_operator = args.binary_operator
        output_path = args.output_file

        if binary_operator == "all":
            binaryOps = ["average", "hadamard", "l1", "l2", "hamming", "cosine", "chisquare"]
        else:
            binaryOps = [binary_operator]

        with open(output_path, 'w') as f:
            for binary_op in binaryOps:
                test_roc = predict(input_folder, graph_name, emb_file, file_type, binary_op)
                f.write("{} {}\n".format(binary_op, test_roc))
