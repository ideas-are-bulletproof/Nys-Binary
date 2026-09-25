import numpy as np


def read_embedding_file(file_path, nodelist=None, file_type="txt"):

    if file_type == "text":

        with open(file_path, 'r') as f:
            # Read the first line
            N, dim = (int(v) for v in f.readline().strip().split())
            if nodelist is None:
                nodelist = list(range(N))
            embs = np.zeros(shape=(len(nodelist), dim), dtype=float) #node2embedding=[[] for _ in range(len(nodelist))]
            mapping = {node: nodeIdx for nodeIdx, node in enumerate(nodelist)}
            # Read embeddings
            for line in f.readlines():
                tokens = line.strip().split()
                if int(tokens[0]) in nodelist:
                    embs[mapping[int(tokens[0])], :] = [float(value) for value in tokens[1:]]
                    # node2embedding[mapping[int(tokens[0])]] = [float(value) for value in tokens[1:]]
        return embs

    elif file_type == "binary":

        def _int2boolean(num):

            binary_repr = []
            for _ in range(8):

                binary_repr.append(True if num % 2 else False )
                num = num >> 1

            return binary_repr[::-1]

        with open(file_path, 'rb') as f:

            num_of_nodes = int.from_bytes(f.read(4), byteorder='little')
            dim = int.from_bytes(f.read(4), byteorder='little')

            if nodelist is None:
                nodelist = list(range(num_of_nodes))
            embs = [[] for _ in range(len(nodelist))]
            mapping = {node: nodeIdx for nodeIdx, node in enumerate(nodelist)}

            dimInBytes = int(dim / 8)

            for i in range(num_of_nodes):
                emb = []
                for _ in range(dimInBytes):
                    emb.extend(_int2boolean(int.from_bytes(f.read(1), byteorder='little')))

                if i in nodelist:
                    embs[mapping[i]] = emb

        return np.asarray(embs, dtype=bool)

    else:

        raise ValueError("Invalid file type!")


# def read_emb_file(file_path, file_type="binary"):
#
#     if file_type == "binary":
#
#         def _int2boolean(num):
#             binary_repr = []
#             for _ in range(8):
#                 binary_repr.append(False if num % 2 else True )
#                 num = num >> 1
#             return binary_repr
#
#         with open(file_path, 'rb') as f:
#
#             num_of_nodes = int.from_bytes(f.read(4), byteorder='little')
#             dim = int.from_bytes(f.read(4), byteorder='little')
#
#             print("{} {}".format(num_of_nodes, dim));
#             dimInBytes = int(dim / 8)
#
#             embs = []
#             for i in range(num_of_nodes):
#                 emb = []
#                 for _ in range(dimInBytes):
#                     emb.extend( _int2boolean(int.from_bytes(f.read(1), byteorder='little')) )
#
#                 embs.append(emb)
#
#             embs = np.asarray(embs, dtype=bool)
#
#     elif file_type == "text":
#
#         with open(file_path, 'r') as fin:
#             # Read the first line
#             num_of_nodes, dim = ( int(token) for token in fin.readline().strip().split() )
#
#             # read the embeddings
#             embs = [[] for _ in range(num_of_nodes)]
#
#             for line in fin.readlines():
#                 tokens = line.strip().split()
#                 vect = [float(v) for v in tokens[1:]]
#                 embs[int(tokens[0])] = vect
#
#             embs = np.asarray(embs, dtype=np.float)
#
#     else:
#         raise ValueError("Invalid method!")
#
#     return embs