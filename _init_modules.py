import os.path as os
import sys

def add_path(path):
    sys.path.insert(0,path)

this_dir = os.dirname(__file__)
add_path(this_dir)

#add util to python_path
util_path = os.join(this_dir,'mxnet-face','util')
extractFeat_path= os.join(this_dir,'mxnet-face')
add_path(extractFeat_path)
add_path(util_path)


