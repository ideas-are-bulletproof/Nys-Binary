from igraph import *
import leidenalg as la
import numpy as np
import sys, random, os
import time
import functools
random.seed(42)
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
            #print(">>" * debug_depth, "single cluster of size", len(clusters[0]))
            # we're done
            return frozenset(vertex_map[x] for x in clusters[0])

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
            return frozenset(tree)
    
    else:
        return frozenset(vertex_map) # Return a 1-depth tree of all vertices.


def leaf_nodes(tree, cur_set=set()):
    if type(tree) in (frozenset, set, tuple, list):
        for node in tree:
            leaf_nodes(node, cur_set)
    else:
        cur_set.add(tree)
    return cur_set

def all_nodes(tree, cur_set=set()):
    cur_set.add(tree)
    if type(tree) in (frozenset, set, tuple, list):
        for node in tree:
            # recursively add child nodes
            all_nodes(node, cur_set)
    return cur_set

# Now we make the forest
def get_forest_of_graphs(G, trees, depth, algorithm, resolution):
    huge_graph = set()
    debug_total_nodes = 0 # G.vcount()
    unique_nodes = set()
    for tree_id in range(trees):
        random.seed(42 + tree_id)

        ## need to update
        # huge_addition, tree = get_tree_of_graphs(node_to_neighbors, layers, quality_factor)
        tree = get_hierarchical_communities( G, depth, algorithm, resolution )
        ## need to update

	    # how many EXTRA nodes were added?
        debug_total_nodes += len(all_nodes(tree))
        
        # put all the nodes into the huge_graph
        print("tree has", len(tree), "immediate children")
        huge_graph.update(tree)
        unique_nodes.update(all_nodes(tree))

        print("\nFINISHED TREE", tree_id + 1)
        print("Total nodes created:", debug_total_nodes, "of which", len(unique_nodes), "are unique\n")

    return frozenset(huge_graph)

# Now we're ready to print this!
@functools.lru_cache(maxsize=None)
def node_name(obj):
    if type(obj) in SCALAR_TYPES:
        return str(obj)
    else:
        # recurrrrr(ecursion)rrrrrrsion
        return '(' + '|'.join(node_name(x) for x in obj) + ')'
        #l = tuple(sorted(leaf_nodes(obj)))
        #if len(l) > 100000:
        #    return '(len={}:{})'.format(len(l), hex(hash(l)))
        #else:
        #    return ','.join(x for x in l)

class PairCollector:
    def __init__(self):
        self.roots_already_seen = set()
        self.collected_hier_pairs = set()
        self.collected_direct_pairs = set()
        self.collected_sib_pairs = set()

    def collect_descendants_of(self, parent, child, sample_ratio = 0.5):
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
                    
                # idea:
                #if len(parent) > 5:
                #    

                # put in the sibling pairs as well
                for child2 in parent:
                    child2_name = node_name(child2)
                    # right here, we *know* it's a direct edge
                    self.collected_direct_pairs.add((child2_name, parent_name))
                    other_children = list(parent)
                    #if len(other_children) >= 10:
                    #    other_children = random.sample(other_children, 10)
                    for other_child in other_children:
                        other_name = node_name(other_child)
                        if child2_name != other_name:
                            self.collected_sib_pairs.add(frozenset({child2_name, other_name}))
        else:
            self.collected_hier_pairs.add((child_name, parent_name))

        if type(child) in (frozenset, tuple, list, set):
            #print("child:", child)
            for grandchild in child:
                self.collect_descendants_of(parent, grandchild)

    def subsample_hier(self):
        indirect = self.collected_hier_pairs - self.collected_direct_pairs
        if len(self.collected_hier_pairs) > 2000000:
        #if len(indirect) > 2*len(self.collected_direct_pairs):
            target = 2000000 - len(self.collected_direct_pairs)
            self.sampled_hier = list(self.collected_direct_pairs) + random.sample(list(indirect), target)
        else:
            self.sampled_hier = self.collected_hier_pairs
           
        #return sampled_hier

        #return self.collected_direct_pairs

    def subsample_sibs(self):
        if len(self.collected_sib_pairs) > 1500000:
            self.sampled_sibs = random.sample(list(self.collected_sib_pairs), 1500000)
        else:
            self.sampled_sibs = self.collected_sib_pairs
        
        #return sampled_sibs
        
    def write_hier(self, filewriter):
        hier_pair_list = sorted(self.sampled_hier, key=node_name)
        for (A, B) in hier_pair_list:
            print(A, B, file=filewriter)

    def write_sibs(self, filewriter):
        sib_pair_list = sorted(tuple(pair) for pair in self.sampled_sibs)
        for (A, B) in sib_pair_list:
            print(A, B, file=filewriter)

    def count_hier(self):
        print("Number of hierarchical pairs:", len(self.collected_hier_pairs))
            
    def count_sibs(self):
        print("Number of sibling pairs:", len(self.collected_sib_pairs),"\n")


def write_forest(tree, filename, trees, depth, q):

    print("\n===== PARSE COMPLETE =====\n")
    
    tree_writer = open("data/Using_Leiden/" + filename + "_t" + str(trees) + "_d" + str(depth) + ".forest", "w")
    sibling_writer = open("data/Using_Leiden/" + filename + "_t" + str(trees) + "_d" + str(depth) + ".siblings", "w")

    collector = PairCollector()
    collector.collect_descendants_of(tree, tree)
    collector.count_hier()
    collector.count_sibs()
    collector.subsample_hier()
    collector.subsample_sibs() 
    
    print("Collected pairs.")
    collector.write_hier(tree_writer)
    collector.write_sibs(sibling_writer)


def get_hypernym_edges(datafile, trees, depth, algorithm = "Leiden", resolution = 0.2, is_weighted=False):

    # extract dataset name from the data_path name
    datafile_abs_path = os.path.abspath(datafile)
    base_name, ext = os.path.splitext(os.path.basename(datafile_abs_path))

    # Read input graph
    G = read_graph(datafile, is_weighted)
    hierarchy_formation_start_time = time.time()
    huge_graph = get_forest_of_graphs(G, trees, depth, algorithm, resolution)
    #collector = PairCollector()
    #collector.collect_descendants_of(huge_graph, huge_graph)
    print("Time taken to form ", base_name, " hierarchy tree:", time.time()-hierarchy_formation_start_time)
    #print("\nCollected pairs.")
    #collector.count_hier()
    #collector.count_sibs()
    #return collector.subsample_hier(), collector.subsample_sibs()
    write_forest(huge_graph, base_name, trees, depth, resolution)
