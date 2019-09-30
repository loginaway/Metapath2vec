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
        type2set:   A dict classifying all node ids into groups by their types.
    '''
    with open(filename, 'r') as fp:
        sent_list = fp.readlines()
        sent_list = [i.strip().split() for i in sent_list]
    
    node2id = {}
    type2set = {type_: set() for type_ in typestr}
    nid = 0
    maxsentlen = 0
    for sent in sent_list:
        if len(sent) > maxsentlen: maxsentlen = len(sent)
        for node in sent:
            prev_id = node2id.get(node, -1)
            if prev_id >= 0:
                continue

            type2set[node[0]].add(nid)
            node2id[node] = nid
            nid += 1
    
    return node2id, type2set, maxsentlen, len(sent_list)

def set_gpu(gpus):
	"""
	Sets the GPU to be used for the run

	Args:
	    gpus:           List of GPUs to be used for the run  
	"""
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    