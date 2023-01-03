import networkx as nx
from graph_tool.all import load_graph
import sys
# sys.path.append("/UnsupervisedGraphTraversal/node2vec")
from node2vec import Node2Vec

# FILES
EMBEDDING_FILENAME = './v_embeddings.emb'
EDGES_EMBEDDING_FILENAME = './e_embeddings.emb'
EMBEDDING_MODEL_FILENAME = './embeddings.model'

# Create a graph - nx version
# graph = nx.fast_gnp_random_graph(n=100, p=0.5)
# Create a graph - graph-tool version
gt_graph_path = "/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/MiceBrainSubgraphs/subgraph_area_Trochlear nucleus.gt"
graph = load_graph(gt_graph_path)
# Precompute probabilities and generate walks
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
# Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
#node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

# Embed
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)

# Embed edges using Hadamard method
from node2vec.edges import HadamardEmbedder

edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

# Look for embeddings on the fly - here we pass normal tuples
edges_embs[('1', '2')]
''' EXPECTED OUTPUT
array([ 5.75068220e-03, -1.10937878e-02,  3.76693785e-01,  2.69105062e-02,
       ... ... ....
       ..................................................................],
      dtype=float32)
'''

# Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
edges_kv = edges_embs.as_keyed_vectors()

# Look for most similar edges - this time tuples must be sorted and as str
edges_kv.most_similar(str(('1', '2')))

# Save embeddings for later use
edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)