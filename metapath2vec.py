# coding: utf-8

import tensorflow as tf
from helper import *
from random import choices

class metapath2vec():
    
    def load_data(self):
        '''
        Load data from files.
        '''
        self.node2id, self.type2set, self.maxsentlen, self.sent_num= getData(self.args.filename, self.args.typestr)
        # Compute max number of windows in a sentence
        self.num_windows = self.maxsentlen
        self.id2node = {self.node2id[key]: key for key in self.node2id}

    def neg_sample(self, cont_list):
        '''
        Conduct negative sampling for cont_list.

        Args;
            cont_list: A 2 dimensional context node id list.
        
        Returns:

        '''
        neg_list = []
        for context in cont_list:
            line = []
            for id_ in context:
                if id_ == -1: 
                    id_set = tuple()
                    avlbl_size = 0
                else:
                    id_set = self.type2set[self.id2node[id_][0]].difference((id_,))
                    avlbl_size = min(len(id_set), self.args.neg_size) 
                line.extend(choices(tuple(id_set), k=avlbl_size)+[-1 for _ in range(self.args.neg_size - avlbl_size)])
            neg_list.append(line)
        return neg_list

    def get_batch(self):
        '''
        Generate a batch of size self.args.batch_size.

        Returns:
            A generator that generate batches, each batch is of the form
            batch <dict> : 
                'neg_ind': Negative samples indexes tensor with size (batch_size, num_windows, neg*2*neighbour_size).
                'cor_ind': Core samples indexes tensor with size (batch_size, num_windows, 1).
                'cont_ind': Context samples indexes tensor with size (batch_size, num_windows, 2*neighbour_size).
        '''
        batch = {}
        sent_count = 0

        batch['neg_ind'] = np.empty(\
            (self.args.batch_size, self.num_windows, self.args.neg_size * 2 * self.args.neighbour_size), dtype=np.int32)
        batch['cor_ind'] = np.empty(\
            (self.args.batch_size, self.num_windows, 1),                                                 dtype=np.int32)
        batch['cont_ind'] = np.empty(\
            (self.args.batch_size, self.num_windows, 2 * self.args.neighbour_size),                      dtype=np.int32)

        patch_list = [-1 for _ in range(self.args.neighbour_size)]
        for line in open(self.args.filename, 'r'):
            if len(line) < 2: continue
            # generate word id list (from a sentence)
            wrd_list = patch_list + [self.node2id[wrd] for wrd in line.strip().split()] + patch_list
            # generate core id list
            core_list = [wrd_list[i] \
                        for i in range(self.args.neighbour_size, len(wrd_list) - self.args.neighbour_size)]
            # generate context id list
            cont_list = [wrd_list[cor_ind - self.args.neighbour_size: cor_ind] + \
                        wrd_list[cor_ind + 1: cor_ind + self.args.neighbour_size + 1] \
                        for cor_ind in range(self.args.neighbour_size, len(wrd_list) - self.args.neighbour_size)]

            # generate negative samples
            neg_list = self.neg_sample(cont_list)  

            batch['cor_ind'][sent_count] = np.reshape(core_list, (len(core_list), 1))
            batch['cont_ind'][sent_count] = cont_list
            batch['neg_ind'][sent_count] = neg_list

            sent_count += 1
            if sent_count == self.args.batch_size:
                sent_count = 0
                yield batch

                batch['neg_ind'] = np.empty(\
                    (self.args.batch_size, self.num_windows, self.args.neg_size * 2 * self.args.neighbour_size), dtype=np.int32)
                batch['cor_ind'] = np.empty(\
                    (self.args.batch_size, self.num_windows, 1),                                                 dtype=np.int32)
                batch['cont_ind'] = np.empty(\
                    (self.args.batch_size, self.num_windows, 2 * self.args.neighbour_size),                      dtype=np.int32)

        batch = {key: batch[key][: sent_count] for key in batch}
        yield batch

    def add_placeholders(self):
        '''
        Add placeholders for metapath2vec.
        '''
        self.negative_ind = tf.placeholder(tf.int32, \
            (None, self.num_windows, self.args.neg_size * 2 * self.args.neighbour_size))
        self.core_ind = tf.placeholder(tf.int32, \
            (None, self.num_windows, 1))
        self.context_ind = tf.placeholder(tf.int32, \
            (None, self.num_windows, 2 * self.args.neighbour_size))

    def create_feed_dict(self, batch):
        '''
        Create feed dict for training.

        Args:
            batch <dict>: Batch generated from next(batch_generator), where batch_generator is 
                the return of self.get_batch().
        
        Returns:
            feed_dict <dict>: the feed dictionary mapping from placeholders to values.
        '''
        feed_dict = {}
        feed_dict[self.negative_ind] = batch['neg_ind']
        feed_dict[self.core_ind] = batch['cor_ind']
        feed_dict[self.context_ind] = batch['cont_ind']
        return feed_dict
    
    def add_embedding(self):
        '''
        Add embedding parameters in the computing.
        '''
        with tf.variable_scope('Embeddings'):
            embed_matrix = tf.get_variable('embed_matrix', 
                [len(self.node2id), self.args.embed_dim], tf.float32, 
                initializer=tf.random_normal_initializer(),
                regularizer=self.regularizer
                )
            padding = tf.get_variable('padding', 
                [1, self.args.embed_dim], tf.float32, 
                initializer=tf.zeros_initializer(), 
                trainable=False
            )
            self.embed_matrix = tf.concat([embed_matrix, padding], axis=0)
    
    def add_model(self):
        '''
        Build metapath2vec structure.
        
        Returns:
            loss: loss of the estimation of the model.
        '''
        with tf.name_scope('Main_Model'):
            neg_embed = tf.nn.embedding_lookup(self.embed_matrix, self.negative_ind)
            core_embed = tf.nn.embedding_lookup(self.embed_matrix, self.core_ind)
            cont_embed = tf.nn.embedding_lookup(self.embed_matrix, self.context_ind)
            
            neg_core = tf.matmul(core_embed, tf.transpose(neg_embed, [0, 1, 3, 2]))
            cont_core = tf.matmul(core_embed, tf.transpose(cont_embed, [0, 1, 3, 2]))

            sec_neg = tf.log(tf.sigmoid(tf.negative(neg_core)))
            sec_cont = tf.log(tf.sigmoid(cont_core))

            objective = tf.reduce_sum(sec_neg) + tf.reduce_sum(sec_cont)
            loss = tf.negative(objective)

            if self.regularizer != None:
                loss += tf.contrib.layers.apply_regularization(self.regularizer, 
                    tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        
        return loss

    def add_optimizer(self, loss):
        '''
        Add optimizer for estimating the parameters.

        Args:
            loss: Model loss from add_model().
        Returns:
            train_op: 
        '''
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
            train_op = optimizer.minimize(loss)

        return train_op
    
    def run_epoch(self, sess, batch):
        '''
        Runs an epoch of training.

        Args:
            sess: tf.Session() object.
            batch: batch generated from next(batch_generator), where batch_generator is 
                the return of get_batch().
        Returns:
            average_loss: Average mini-batch loss on this epoch.
        '''
        loss_list = []
        current_sent_num = 0

        for batch_size in 


            


    def __init__(self, args):
        '''
        Initialize metapath2vec model with args.
        
        Args:
            args: An instance of class argparse. Details are in if __name__ == '__main__' clause.
        '''
        self.args = args
        self.load_data()
        if self.args.l2 != 0: 
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.args.l2)
        else: 
            self.regularizer = None









if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Metapath2Vec')

    parser.add_argument('-file',        dest='filename',    default='walks.txt',    help='The random walks filename')
    parser.add_argument('-embed_dim',   dest='embed_dim',   default=64,   type=int, help='The length of latent embedding')
    parser.add_argument('-n',      dest='neighbour_size',   default=7,    type=int, help='The neighbourhood size k')
    parser.add_argument('-epoch',       dest='epoch',       default=10,   type=int, help='Num of iterations for heterogeneous skipgram')
    parser.add_argument('-types',       dest='typestr',     default='a',  type=str, help='Specify types occurring in the data')
    parser.add_argument('-batch',       dest='batch_size',  default=4,   type=int, help='The number of the data used in each iter')
    parser.add_argument('-neg',         dest='neg_size',    default=5,    type=int, help='The size of negative samples')
    parser.add_argument('-gpu',         dest='gpu',         default='0',            help='Run the model on gpu')
    parser.add_argument('-l2',          dest='l2',        default=0.001,  type=float,help='L2 regularization scale (default 0.001)')

    args = parser.parse_args()
    set_gpu(args.gpu)

    model = metapath2vec(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.75

    # with tf.Session(config=config) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     model.fit()
    batch = model.get_batch()
    # get = next(batch)
    # print(get, [i.shape for i in get.values()])
    model.add_embedding()
    model.add_placeholders()
    print(model.embed_matrix)
    model.add_model()
    
