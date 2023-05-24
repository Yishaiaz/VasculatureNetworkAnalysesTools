import os
import sys
import shutil
import math
import warnings
from typing import *
from enum import Enum
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graph_tool.all as gt
import igraph as ig
import chart_studio.plotly as py
from plotly.offline import iplot
import plotly.graph_objs as go


def convert_graph_tool_to_igraph(_g: gt.Graph,
                                 include_attributes: bool = False,
                                 **kwargs) -> ig.Graph:
    # todo: 1 - add the ability to directly save the new graph instance.
    # todo: option: ig_graph.write("graph_file.graphml", format="graphml")

    all_gt_edges = [(int(_edge.source()), int(_edge.target())) for _edge in _g.edges()]
    ig_graph = ig.Graph(all_gt_edges, directed=_g.is_directed())
    if _g.num_edges() == ig_graph.ecount():
        warnings.warn(f"Number of edges is not identical after conversion! # in gt graph: {_g.num_edges()}!= # in igraph graph: {ig_graph.ecount()}")
    if _g.num_vertices() == ig_graph.vcount():
        warnings.warn(f"Number of vertices is not identical after conversion! # in gt graph: {_g.num_vertices()}!= # in igraph graph: {ig_graph.vcount()}")

    for prop in _g.vertex_properties.keys():
        ig_graph.vs[prop] = _g.vertex_properties[prop].get_array()
    for prop in _g.edge_properties.keys():
        ig_graph.es[prop] = _g.edge_properties[prop].get_array()

    return ig_graph


if __name__ == '__main__':
    gt_graph_path = "/Users/yishaiazabary/PycharmProjects/University/BrainVasculatureGraphs/Data/MiceBrainSubgraphs/subgraph_area_amygdalar capsule.gt"
    gt_graph = gt.load_graph(gt_graph_path)
    convert_graph_tool_to_igraph(gt_graph)