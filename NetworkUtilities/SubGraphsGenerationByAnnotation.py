import json
import os
import shutil
import datetime
# from graph_tool.all import *
import warnings
"""
exit codes:
1 - unknown error
111 - could not find configuration file 
1112 - could not parse configuration file as json
112 - could not import clear map or compile it
113 - could not load entire graph data 
"""

PATH_TO_SUB_GRAPHS_DIR = 'Data/subgraphsByAnnotationLabels'

try:
    with open('job_conf_file.txt', 'r') as f:
        print("FOUND FILE!, the path to the ssd dir is:")
        configuration_file_dict = f.readlines()[0]
    ssd_path_to_dir = configuration_file_dict
    print(ssd_path_to_dir)
except FileNotFoundError as e:
    print("could not find file, but here is the cwd listdir")
    print(f"CWD:{os.getcwd()}")
    print(os.listdir(os.getcwd()))
    exit(111)
except Exception as e:
    print(f'could not parse configuration file json: {e}')
    exit(1112)
print(f"SSD dir path: {ssd_path_to_dir}")

try:
    import sys
    sys.path.insert(0, '/home/yishaiaz/ClearMap2')
    print("Attempting to compile ClearMap")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import ClearMap.Compile
    print("Finished to compile ClearMap")
except ImportError as e:
    print(f"could not import ClearMap")
    exit(112)

# load graph:

entire_graph_path = os.path.join(ssd_path_to_dir.strip(), 'data_graph.gt')

# try:
#     print("Attempting to load entire graph")
#     entire_graph = load_graph(entire_graph_path)
# except FileNotFoundError as e:
#     print(f"File not found at: {entire_graph_path}")
#     exit(113)
# except Exception as e:
#     print(f"Uknown expection occured: {e}")
#     exit(1)

# read vertex annotations
try:
    print("Attempting to load clearmap annotation module")
    import ClearMap.Alignment.Annotation as ano
    # from ClearMap.Environment import *
    from ClearMap.Analysis import *
    # print(dir(Graphs))
    # print(dir(Graphs.GraphGt))
except ImportError as e:
    print(f"could not import annotation module from clearmap {e}")
    exit(1)

try:
    print("attempting to load entire graph via clearmap API")
    entire_graph = Graphs.GraphGt.load(entire_graph_path)
    print("Finished loading entire graph via clearmap API")
except Exception as e:
    print(f"ERROR:\n{e}")
    exit(1)
try:
    print("Attempting to generate subgraphs by vertex annotation labels")
    curr_date = datetime.datetime.now().strftime('%%m_%d_%y')
    #############################
    # for a range of hierarchies:
    # all_vs_annotations = entire_graph.vertex_annotation()
    # converted_labels_by_hierarchy = {}
    # curr_run_path_to_dir = os.path.join(PATH_TO_SUB_GRAPHS_DIR, curr_date)
    # os.makedirs(curr_run_path_to_dir, exist_ok=True)
    # for hierarchy_lvl in [None] + list(range(1, 7)):
    #     print(f"processing hierarchy levels: {hierarchy_lvl}")
    #     curr_run_path_to_dir_hierarchy = os.path.join(curr_run_path_to_dir, f"hierarchy_{hierarchy_lvl if hierarchy_lvl is not None else 'none'}")
    #     os.makedirs(curr_run_path_to_dir_hierarchy, exist_ok=True)
    #     converted_all_vs_annotations = ano.convert_label(all_vs_annotations, key='order', value='order',
    #                                                      level=hierarchy_lvl)
    #     print(f"#converted labels in hierarchy level = {hierarchy_lvl} is {len(converted_all_vs_annotations)}")
    #
    #     vertex_filter = converted_all_vs_annotations == hierarchy_lvl
    #     subgraph_by_label = entire_graph.sub_graph(vertex_filter=vertex_filter)
    #     print(f"size of in hierarchy level = {hierarchy_lvl} is {subgraph_by_label.n_vertices()}")
    #     gt_subgraph_by_label = subgraph_by_label.base
    #     converted_labels_by_hierarchy[hierarchy_lvl] = converted_all_vs_annotations
    #     gt_subgraph_by_label_path = os.path.join(curr_run_path_to_dir_hierarchy, f"subgraph.gt")
    #     gt_subgraph_by_label.save(gt_subgraph_by_label_path)
    #
    # curr_run_path_to_dir_summation_path = os.path.join(curr_run_path_to_dir, curr_date,
    #                                                    "annotations_by_hierarchy_levels.txt")
    # with open(curr_run_path_to_dir_summation_path, 'w') as f:
    #     json.dump(converted_labels_by_hierarchy, f)

    #############################
    # for a single hierarchy:
    hierarchy_lvl = 6
    all_vs_annotations = entire_graph.vertex_annotation()
    curr_run_path_to_dir = os.path.join(PATH_TO_SUB_GRAPHS_DIR, curr_date)
    # os.makedirs(curr_run_path_to_dir, exist_ok=True)
    converted_all_vs_annotations = ano.convert_label(all_vs_annotations, key='order', value='order',
                                                         level=hierarchy_lvl)
    print(converted_all_vs_annotations)
    hierarchy_names = [ano.find(x, key='order')['name'] for x in converted_all_vs_annotations]
    print(hierarchy_names)


except Exception as e:
    print(e)
    exit(1)






