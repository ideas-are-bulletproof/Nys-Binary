#ifndef GRAPH_H
#define GRAPH_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;



class Graph {

private:
    bool _directed;
    vector <vector <pair<unsigned int, double>>> _adjList;
    unsigned int _numOfNodes;
    unsigned int _numOfEdges;


public:
    Graph(bool directed);
    Graph();
    ~Graph();

    //vector <int> getCommonNeighbours(int u, int v);
    void readEdgeList(string file_path, bool verbose);
    void writeEdgeList(string file_path, bool weighted);
    unsigned int getNumOfNodes();
    unsigned int getNumOfEdges();
    template <typename T>
    vector <vector <pair<unsigned int, double>>> getAdjList();

};

template <typename T>
vector <vector <pair<unsigned int, double>>> Graph::getAdjList() {

    return this->_adjList;

}


#endif //GRAPH_H