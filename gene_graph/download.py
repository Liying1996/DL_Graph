import networkx as nx
import os
import academictorrents as at

def load_data():
    
    savefile = os.path.join("/mnt/double-graph/gene_graph/GeneManiaGraph.adjlist.gz")
    
    if os.path.isfile(savefile):
        print(" loading from cache file" + savefile)
        nx_graph = nx.read_adjlist(savefile)
        return nx_graph
    else:
    
        nx_graph = nx.OrderedGraph(
            nx.readwrite.gpickle.read_gpickle(at.get(at_hash="5adbacb0b7ea663ac4a7758d39250a1bd28c5b40", datastore='/mnt/double-graph/gene_graph/')))
        
        print(" writing graph...")
        nx.write_adjlist(nx_graph, savefile)

def cal_day(c, y, m, d):
    w = (y + int(y/4) + int(c/4) - 2*c + 2*m + int(3*(m+1)/5) + d + 1)
    return w % 7

cal_day(20, 21, 12, 16)