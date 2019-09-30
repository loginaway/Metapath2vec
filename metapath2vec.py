# coding: utf-8

import tensorflow as tf
from helper import *

class metapath2vec():
    
    def load_data(self):
        '''
        Load data from files.

        Returns:

        '''


    def __init__(self, args):
        '''
        Initialize metapath2vec model with args
        
        Args:
            args: An instance of class argparse.
        '''








if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Metapath2Vec')

    parser.add_argument('-file',        dest='filename',    default='walks.txt',    help='The random walks filename')
    parser.add_argument('-embed_dim',   dest='embed_dim',   default=64,   type=int, help='The length of latent embedding')
    parser.add_argument('-n',      dest='neighbour_size',   default=7,    type=int, help='The neighbourhood size k')
    parser.add_argument('-epoch',       dest='epoch',       default=10,   type=int, help='Num of iterations for heterogeneous skipgram')
    parser.add_argument('-types',       dest='typestr',     default='a',  type=str, help='Specify types occurring in the data')
    parser.add_argument('-batch',       dest='batch_size',  default=64,   type=int, help='The number of the data used in each iter')
    parser.add_argument('-neg',         dest='neg_size',    default=5,    type=int, help='The size of negative samples')
    parser.add_argument('-gpu',         dest='gpu',         default='0',            help='Run the model on gpu')

    args = parser.parse_args()
    set_gpu(args.gpu)

    model = metapath2vec(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.75

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.fit()
    
