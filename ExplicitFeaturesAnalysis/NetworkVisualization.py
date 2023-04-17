
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
from graph_tool import draw
from ExplicitFeaturesExtractor import ExplicitFeatures as fe


def draw_graph_with_color_edges_by_edge_attribute(_g: gt.Graph,
                                                  color_attribute_name: str = None,
                                                  size_attribute_name: str = None,
                                                  edge_or_vertex: str = 'edge',
                                                  # color_map: Union[Callable, Dict],
                                                  dir_to_save_graph_draw: Union[os.PathLike, str]=".",
                                                  graph_draw_name: str = None,
                                                  **kwargs):
    output_fname = graph_draw_name if graph_draw_name is not None else "test_output_for_function_draw_graph_with_color_edges_by_edge_attribute.pdf"
    if any([not output_fname.endswith(_fmt) for _fmt in kwargs.get("allowed_formats_of_output_file", ('.png', '.pdf'))]):
        warnings.warn(f"Output file format must be one of: [{kwargs.get('allowed_formats_of_output_file', ('.png', '.pdf'))}], but got: {output_fname}.\nsaving in PDF format")
        output_fname += '.pdf'

    key_type = 'e' if edge_or_vertex.lower() == 'edge' else 'v'
    output_path = os.path.join(dir_to_save_graph_draw, output_fname)
    print(f"saving {graph_draw_name} @ {output_path}")
    color_prop_map = None
    if color_attribute_name is not None:
        attribute_array = _g.properties[(key_type, color_attribute_name)].a
        max = attribute_array.max()
        min = attribute_array.min()
    #    norm_attribute_array = ((attribute_array.max() - attribute_array)/(attribute_array.max()-attribute_array.min()))**10
        norm_attribute_array = attribute_array
        color_attribute_array = np.zeros(shape=(len(norm_attribute_array), 4))
        color_attribute_array[:, 3] = 1
        for idx in range(len(norm_attribute_array)):
            if norm_attribute_array[idx] == -1:
                color_attribute_array[idx, 2] = 1
            if norm_attribute_array[idx] == 0:
                color_attribute_array[idx, 3] = 0
            if norm_attribute_array[idx] == 1:
                color_attribute_array[idx, 1] = 1
            if norm_attribute_array[idx] == 3:
                color_attribute_array[idx, 0] = 1

#        color_attribute_array[:, 0] = 1/2+norm_attribute_array/2
        color_prop_map = _g.new_property(key_type, 'vector<double>', vals=color_attribute_array)

    size_prop_map = None
    if size_attribute_name is not None:
        attribute_array = _g.properties[(key_type, size_attribute_name)].a
        size_attribute_array = attribute_array.copy().astype(float)
        for idx in range(len(size_attribute_array)):
            if size_attribute_array[idx] > 1:
                size_attribute_array[idx] = 1

        size_attribute_array = (size_attribute_array * 2)
        size_attribute_array += 1
        size_attribute_array /= 2
        size_prop_map = _g.new_property(key_type, 'double', vals=size_attribute_array)

    draw_kwargs = {
        f'{edge_or_vertex}_{"" if key_type=="e" else "fill_"}color': color_prop_map,
        f'{edge_or_vertex}_{"pen_width" if key_type=="e" else "size"}': size_prop_map,
    }

    draw.graph_draw(_g, pos=_g.vp['coordinates'], output=output_path, **draw_kwargs)
#    gt.interactive_window(_g, pos=_g.vp['coordinates'],
#                  output=output_path,
#                  **draw_kwargs)

# def draw_graph_in_3d(_g: gt.Graph):
#

if __name__ == '__main__':
    gbm_graph_fpath = "/Users/leahbiram/Desktop/vasculature_data/firstGBMscanGraph.gt"
    g = gt.load_graph(gbm_graph_fpath)
    gbm_graph = fe.preprocess_graph(g)

    coo = g.vp['coordinates'].fa

    file_path = "/Users/leahbiram/Desktop/vasculature_data/firstGBMscanGraph.csv"
    df = pd.read_pickle(file_path)
    p_list = df["predictions"].values[:, -1]
    r_list = gbm_graph.vertex_properties["artery_binary"].fa

    gbm_graph.vertex_properties["predictions"] = gbm_graph.new_vertex_property("int")
    for i, v in enumerate(gbm_graph.vertices()):
        gbm_graph.vertex_properties["predictions"][v] = p_list[i]

    correctness = p_list
    for i in range(len(p_list)):
        if p_list[i] == 0 and r_list[i] == 1:
            correctness[i] = 2
        if p_list[i] == 1 and r_list[i] == 0:
            correctness[i] = 3
   # correctness = [2 if p==0 and r==1 else p for p in p_list for r in r_list]

    gbm_graph.vertex_properties["correct_preds"] = gbm_graph.new_vertex_property("int")
    for i, v in enumerate(gbm_graph.vertices()):
        gbm_graph.vertex_properties["correct_preds"][v] = correctness[i]

    print(f"properties in loaded graph:  {gbm_graph.list_properties()}")
    draw_graph_with_color_edges_by_edge_attribute(
        _g=gbm_graph,
        edge_or_vertex='vertex',
        color_attribute_name="correct_preds",
        size_attribute_name="correct_preds",
        graph_draw_name="correctness_predictions_vertex_gbm_graph_with_color_norm_immune_intensity_width_immune_presence"
    )
