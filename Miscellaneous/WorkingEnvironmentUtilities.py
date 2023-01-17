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


def get_project_root_dir():
    cwd = os.getcwd()
    cwd_spl = list(filter(lambda x: x!='', cwd.split(os.sep)))
    proj_root_dir_path_lst = [cwd_spl[0]]
    curr_dir = os.path.join(os.sep, *proj_root_dir_path_lst)
    for dir_in_path in cwd_spl[1:]:
        if "MotifsExplorationCode" in os.listdir(curr_dir):
                # stop here
            return curr_dir
        else:
            proj_root_dir_path_lst.append(dir_in_path)
            curr_dir = os.path.join(os.sep, *proj_root_dir_path_lst)
    return curr_dir

if __name__ == '__main__':
    print(get_project_root_dir())
