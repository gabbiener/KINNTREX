# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:35:22 2023

@author: biener
"""

import os
import numpy as np
import mAECell_Utils as utils
import re

File_Names = utils.fOpen(Title = 'Load Data File')

# directory/folder path
dir_path   = os.path.dirname(File_Names)
Data_fName = os.path.basename(File_Names)
dir_path   = os.path.dirname(dir_path)

# list to store files
res = []

# Iterate directory
for file_path in os.listdir(dir_path):
    # check if current file_path is a file
    Dir = os.path.join(dir_path, file_path)
    if os.path.isdir(Dir) and 'Run' in Dir:
        File_toAdd = os.path.join(Dir, Data_fName).replace("/","\\")
        if os.path.exists(File_toAdd):
        
            # add filename to list
            res.append(File_toAdd)
File_Names_File = utils.fSaveAs(Title='Save File Names')
res.sort(key=lambda f: int(re.sub('\D', '', f)))
np.savetxt(File_Names_File, np.array(res), delimiter=',', fmt='%s')