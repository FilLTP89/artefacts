# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import json
from fnmatch import fnmatch
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Get train and test IDs
def allDivs(n):
    div = []
    for i in range(1, n):
        if((n % i) == 0):
            div.append(i)
    return div

def getPath(database="./",folder="./"):
    search_folder = os.path.join("{:>s}".format(database),
        "{:>s}".format(folder))
    filename_list = [os.path.join("{:>s}".format(search_folder),"{:>s}".format(f)) for f in os.listdir(search_folder) if fnmatch(f, '*.raw')]

    filename_list = sorted(filename_list)
    return filename_list

def saveHistory(filename,history):
    out_file = open(filename,"w")
    json.dump(history.history, out_file, indent = 6)
    out_file.close()
    return