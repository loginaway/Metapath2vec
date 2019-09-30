# coding: utf-8

import numpy as np, argparse, os

def getData(filename, typestr):
    '''
    Python version of data reader.

    Args:
        filename:   The name of the target random walks text file.
        typestr:    A string with each character representing one type of node.

    Returns:
        node2id:    A dict mapping node to its id in embeddings.
        type2set:   A dict classifying all nodes into groups by their types.
    '''
    with open(filename, 'r') as fp:
        sent_list = fp.readlines()
        sent_list = [i.strip().split() for i in sent_list]
    
    node2id = {}
    type2set = {type_: set() for type_ in typestr}
    nid = 0
    for sent in sent_list:
        for node in sent:
            type2set[node[0]].add(node)
            node2id[node] = nid
            nid += 1
    
    return node2id, type2set

def set_gpu(gpus):
	"""
	Sets the GPU to be used for the run

	Args:
	    gpus:           List of GPUs to be used for the run  
	"""
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    