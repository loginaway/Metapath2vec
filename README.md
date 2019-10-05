# Metapath2vec




This is a Tensorflow implementation of [Metapath2vec: *Scalable Representation Learning for Heterogeneous Networks*](https://www3.nd.edu/%7Edial/publications/dong2017metapath2vec.pdf).     

The author of the paper also released [the source code and dataset](https://ericdongyx.github.io/metapath2vec/m2v.html). If you want to replicate the experimental results, please use the author's code.    



#### For any clarification, comments or suggestions please create an issue or contact loginaway@gmail.com.

## Dependencies

​	-> Dependencies can be installed using `requirements.txt`.

​	-> Install the dependencies by `pip3 install -r requirements.txt`.

## Dataset

​	-> The input for this code is a text file with each line being a *meta-walk*, an example is put below.

​	`a1312 b1235 c12 d0981 c1563 b123 a1000 b1222 …`

​	where each node is represented by its type (an alphabet) and its number within the typeset. Note that this implementation will generate embeddings for all the nodes that occur in the text file.

​	-> As an example, you can refer to file `walks.txt`.

## Train Metapath2vec embeddings

​	-> To start training, you can run

​	`python metapath2vec.py -file walks.txt -embed_dim 64 -epoch 10 -types ac -outname embeddings`

​	where `-file` specifies the dataset, `-types ac` gives a description of all the types appearing in the dataset (i.e. nodes start with 'a' or 'c').

​	-> Our code has many other options. Run `python metapath2vec.py -h` for more information.

## Output

​	-> The generated embeddings will be put in `./output/`. 



