"""
IDEAS:
1. a siamese like GNN where the input is 2 subgraphs and the model
    predicts whether the subgraphs are similar.
    Similarity will be determined only by the presence of the bio markers.
    1.1 If successful, the model's hidden states can be used for an upstream task to
        classify single subgraphs.
2. Find an architecture that uses the edge attributes (length+radii).
3. For each node in the k-hop subgraph, calculate its distance to the center node
    and insert it as a node attribute.
"""