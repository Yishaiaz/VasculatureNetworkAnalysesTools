import os
import numpy as np
from SubgraphFeatures import ExportFeatureVectors as efv

dir_path = "/Users/leahbiram/Desktop/vasculature_data/SubGraphsByRegion/"
cancer_cells_array_path = "/Users/leahbiram/Desktop/vasculature_data/CancerCellsArray.npy"
cancer_cells_info = np.load(cancer_cells_array_path)
clist = [canc[-1] for canc in cancer_cells_info]
regions = list(set(clist))
for filename in os.listdir(dir_path):
    print("extracting features from: " + filename)
    region_name = filename.removesuffix(".gt").removeprefix("subgraph_area_")
    if region_name in regions:
        efv.main(dir_path + filename, region_name, cancer_cells_array_path)
        print("finished extracting features in " + filename)


# 107 of total of 547 cancer regions, and total 112 files
