/* Graph Version 0.3.0 */
#include "Graph.h"

Graph::Graph(bool directed) {

    _numOfNodes = 0;
    _numOfEdges = 0;
    _directed = directed;

    if(directed) {
        cout << "Not implemented for directed graphs yet!" << endl;
        throw;
    }

}

Graph::Graph():Graph(0) {

}

Graph::~Graph() {

}

void Graph::readEdgeList(string file_path, bool verbose) {

    fstream fs(file_path, fstream::in);
    if(!fs.is_open()) {

        cout << "An error occurred during reading the file!" << endl;
        throw;

    } else {

        if(verbose) {
            cout << "+ The given edge list file is being read." << endl;
            cout << "(Note: It is assumed that node labels start from 0 to N-1 and the graph is undirected.)" << endl;
        }

        unsigned int u, v;
        double weight;
        unsigned int maxNodeLabel = 0;
        unsigned int minNodeLabel = numeric_limits<unsigned int>::max();
        string token;
        int count;
        string line;
        vector <pair<unsigned int, pair<unsigned int, double>>> tempEdgeContainer;


        while (std::getline(fs, line)) {
            stringstream linestream(line);
            // Read the edges
            //linestream >> u >> v >> weight;

            count = 0;
            weight = 1.0; // Default value
            while(getline(linestream, token, ' ')) {
                if(count == 0) {
                    u = (unsigned int) stoul(token);
                } else if (count == 1) {
                    v = (unsigned int) stoul(token);
                } else if(count == 2){
                    weight = stod(token);
                } else {
                    cout << "Splitting problem!" << endl;
                    throw;
                }
                count++;


            }

            //cout << "=" << weight << endl;
            // If no weight is given, it is assumed that it is equal to 1.
            //if (weight == 0)
            //    weight = 1.0;

            // Add the edge to the list
            pair <unsigned int, pair <unsigned int, double>> edge = make_pair(u, make_pair(v, weight));
            tempEdgeContainer.push_back(edge);

            // Find the maximum node label
            if (u > maxNodeLabel) { maxNodeLabel = u; }
            if (v > maxNodeLabel) { maxNodeLabel = v; }
            // Find the minimum node label
            if (u < minNodeLabel) { minNodeLabel = u; }
            if (v < minNodeLabel) { minNodeLabel = v; }

        }
        fs.close();

        // Set the number of nodes and number of edges
        _numOfEdges = (unsigned int) tempEdgeContainer.size();
        _numOfNodes = maxNodeLabel - minNodeLabel + 1;

        /*
        // Store the edges in adjacency list format.
        _adjList.resize(_numOfNodes);
        while(!tempEdgeContainer.empty()) {
            pair <unsigned int, pair <unsigned int, double>> triplet = tempEdgeContainer.back();
            pair <unsigned int, double> &tempEdge = triplet.second;
            if(triplet.first <= tempEdge.first)
                _adjList[triplet.first].push_back(tempEdge);
            tempEdgeContainer.pop_back();
        }

        // Sort the nodes in the adjacency list for efficiency.
        for(unsigned int i=0; i<_numOfNodes; i++)
            sort(_adjList[i].begin(), _adjList[i].end());

        // Check if the graph contains a node of label 0
        if(minNodeLabel != 0) {
            cout << "\t- (Warning) It does not contain the node of label '0'" << endl;
        }

        if(verbose) {
            cout << "\t- It contains " << _numOfNodes << " nodes." << endl;
            cout << "\t- It contains " << _numOfEdges << " edges." << endl;
        }
         */

        // Store the edges in adjacency list format.
        _adjList.resize(_numOfNodes);
        while(!tempEdgeContainer.empty()) {
            pair <unsigned int, pair <unsigned int, double>> triplet = tempEdgeContainer.back();
            pair <unsigned int, double> &tempEdge = triplet.second;
            if(triplet.first <= tempEdge.first)
                _adjList[triplet.first].push_back(tempEdge);
            else
                _adjList[get<0>(tempEdge)].push_back(pair<unsigned int, double>(triplet.first, tempEdge.second));
            tempEdgeContainer.pop_back();

        }

        // Sort the nodes in the adjacency list for efficiency.
        for(unsigned int i=0; i<_numOfNodes; i++)
            sort(_adjList[i].begin(), _adjList[i].end());

        // Check if the graph contains a node of label 0
        if(minNodeLabel != 0) {
            cout << "\t- (Warning) It does not contain the node of label '0'" << endl;
        }

        if(verbose) {
            cout << "\t- It contains " << _numOfNodes << " nodes." << endl;
            cout << "\t- It contains " << _numOfEdges << " edges." << endl;
        }

    }

}

void Graph::writeEdgeList(string file_path, bool weighted) {

    fstream fs(file_path, fstream::out);
    if(fs.is_open()) {
        int u, v;
        int maxNodeId=0, minNodeId = 0;

        for(int node=0; node <_numOfNodes; node++) {

            for(int j=0; j < _adjList[node].size(); j++) {

                fs << node << " " << _adjList[node][j].first;
                if(weighted)
                    fs << " " << _adjList[node][j].second;
                fs << "\n";

            }

        }
        fs.close();

    } else {
        cout << "An error occurred during opening the file!" << endl;
    }

}

/*
vector <int> Graph::getCommonNeighbours(int u, int v) {

    vector <int> common_node_list;
    long unsigned int u_nb_inx= 0, v_nb_inx=0;

    while(u_nb_inx<adjlist[u].size() && v_nb_inx<adjlist[v].size()) {
        if (adjlist[u][u_nb_inx] < adjlist[v][v_nb_inx]) {

            u_nb_inx++;

        } else {

            if(adjlist[u][u_nb_inx] == adjlist[v][v_nb_inx]) {
                common_node_list.push_back(adjlist[u][u_nb_inx]);
            }

            v_nb_inx++;
        }

    }

    return common_node_list;
}

double Graph::getClusteringCoefficient(int v, int u) {

    long unsigned int u_nb_inx= 0, v_nb_inx=0;
    double common_count = 0;

    while(u_nb_inx<adjlist[u].size() && v_nb_inx<adjlist[v].size()) {
        if (adjlist[u][u_nb_inx] < adjlist[v][v_nb_inx]) {

            u_nb_inx++;

        } else {

            if(adjlist[u][u_nb_inx] == adjlist[v][v_nb_inx]) {
                common_count += 1.0;
            }

            v_nb_inx++;
        }

    }

    if(adjlist[v] > adjlist[u]) {

        return common_count / (double)adjlist[u].size();

    } else {

        return common_count / (double)adjlist[v].size();
    }

}


void Graph::vector2Adjlist(bool directed) {

    adjlist.resize(num_of_nodes);

    for(unsigned int j=0; j<num_of_edges; j++) {
        adjlist[edges[j][0]].push_back(edges[j][1]);
        if( !directed ) {
            adjlist[edges[j][1]].push_back(edges[j][0]);
        }
    }

    // Sort the nodes in the adjacency list
    for(unsigned int i=0; i<num_of_nodes; i++)
        sort(adjlist[i].begin(), adjlist[i].end());

}
 */

/*
void Graph::readGraph(string file_path, string filetype, bool directed) {

    if( filetype == "edgelist" ) {

        readEdgeList(file_path, 1);
        // Convert edges into adjacent list
        vector2Adjlist(directed);

    } else {
        cout << "Unknown file type!" << endl;
    }

}
*/


unsigned int Graph::getNumOfNodes() {

    return _numOfNodes;
}

unsigned int Graph::getNumOfEdges() {

    return _numOfEdges;
}



/*
void Graph::printAdjList() {

    for(unsigned int i=0; i<num_of_nodes; i++) {
        cout << i << ": ";
        for(unsigned int j=0; j<adjlist[i].size(); j++) {
            cout << j << " ";
        }
        cout << endl;
    }

}
*/


/*
vector <int> Graph::getDegreeSequence() {

    vector <int> degree_seq(num_of_nodes);
    //degree_seq.resize(num_of_nodes);
    for(unsigned int i=0; i<num_of_nodes; i++) {
        degree_seq[i] = (unsigned int) adjlist[i].size();
    }



    return degree_seq;

}
 */