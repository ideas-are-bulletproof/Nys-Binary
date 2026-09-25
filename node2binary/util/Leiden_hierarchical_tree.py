from igraph import *
import leidenalg as la
import numpy as np
import sys, random, os
random.seed(42)
import time
SCALAR_TYPES = (str, int, np.int32, np.int64)

def read_graph(input_file, is_weighted):
    '''
    Reads the input network in igraph.
    '''

    G = Graph.Read_Ncol(input_file, names=True, directed=False, weights=is_weighted)

    return G
    
    
def get_hierarchical_communities( G, max_depth, algorithm, resolution, vertex_map= None, debug_depth =1 ):
    
    if vertex_map == None:
        vertex_map = [v['name'] for v in G.vs]
    #print("vertex_map:", vertex_map)

    #print(">>" * (debug_depth - 1) + ">", "vertex_map =", vertex_map)

    if debug_depth < max_depth:
        
        clusters = la.find_partition(G, la.ModularityVertexPartition)
        
        if len(clusters) == 1:
            #print(">>" * debug_depth, "single cluster", clusters[0])
            # we're done
            return tuple(vertex_map[x] for x in clusters[0])

        else:
            tree = []
            for cluster in clusters:
                # debug
                #print(">>" * debug_depth, "clustering", cluster)
                tree.append(get_hierarchical_communities(
                    G.induced_subgraph(cluster),
                    max_depth,
                    algorithm, resolution,
                    [vertex_map[x] for x in cluster],
                    debug_depth + 1))
            return tuple(tree)
    
    else:
        return tuple(vertex_map)
   

# Now we're ready to print this!
def node_name(obj):
    if type(obj) in SCALAR_TYPES:
        return str(obj)
    else:
        # recurrrrr(ecursion)rrrrrrsion
        return '(' + '|'.join(node_name(x) for x in obj) + ')'

class PairCollector:
    def __init__(self):
        self.roots_already_seen = set()
        self.collected_hier_pairs = set()
        self.collected_sib_pairs = set()

    def collect_descendants_of(self, parent, child):
        parent_name = parent if type(parent) in SCALAR_TYPES else node_name(parent)
        child_name = child if type(child) in SCALAR_TYPES else node_name(child)

        #print("collecting", parent, child)
        if parent == child:
            # if we've already recursed (parent)'s subtree,
            # there's no need to do it again!
            if parent in self.roots_already_seen:
                return
            self.roots_already_seen.add(parent)

            # passing (parent, parent) means doing all (parent)'s child nodes
            if type(parent) in (frozenset, tuple, list, set):
                for child2 in parent:
                    self.collect_descendants_of(child2, child2)

                # put in the sibling pairs as well
                for child2 in parent:
                    child2_name = node_name(child2)
                    for other_child in parent:
                        if child2 != other_child:
                            other_name = node_name(other_child)
                            self.collected_sib_pairs.add(frozenset({child2_name, other_name}))
        else:
            self.collected_hier_pairs.add((child_name, parent_name))

        if type(child) in (frozenset, tuple, list, set):
            #print("child:", child)
            for grandchild in child:
                self.collect_descendants_of(parent, grandchild)

    def write_hier(self, filewriter):
        hier_pair_list = sorted(self.collected_hier_pairs, key=node_name)
        for (A, B) in hier_pair_list:
            print(A, B, file=filewriter)

    def write_sibs(self, filewriter):
        sib_pair_list = sorted(tuple(pair) for pair in self.collected_sib_pairs)
        for (A, B) in sib_pair_list:
            print(A, B, file=filewriter)

    def count_hier(self):
        print("Number of hierarchical pairs:", len(self.collected_hier_pairs))
            
    def count_sibs(self):
        print("Number of sibling pairs:", len(self.collected_sib_pairs),"\n")


def write_tree(tree, filename, depth, algorithm, q):

    print("\n===== PARSE COMPLETE =====\n")
    
    tree_writer = open("data/Using_Leiden/" + filename + "_d" + str(depth) + ".tree", "w")
    sibling_writer = open("data/Using_Leiden/" + filename + "_d" + str(depth) + ".siblings", "w")

    collector = PairCollector()
    collector.collect_descendants_of(tree, tree)
    collector.count_hier()
    collector.write_hier(tree_writer)
    collector.count_sibs()
    collector.write_sibs(sibling_writer)


def get_hypernym_edges(datafile, depth, algorithm = "Leiden", resolution = 0.2, is_weighted=False):

    # extract dataset name from the data_path name
    datafile_abs_path = os.path.abspath(datafile)
    base_name, ext = os.path.splitext(os.path.basename(datafile_abs_path))

    # Read input graph
    G = read_graph(datafile, is_weighted)
    hierarchy_formation_start_time = time.time()
    tree = get_hierarchical_communities(G, depth, algorithm, resolution)
    print("Time taken to form", basename, "hierarchy tree:", time.time()-hierarchy_formation_start_time)
    write_tree(tree, base_name, depth, algorithm, resolution)
