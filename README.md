#DeepLearning

in the course of deep learning we learned the basics of deep learning and learned different algorithms  Which are CNN ,  RNN , LSTM , NN for find the sentence most similar to given query  and more
In the four files You see in this repository there are 4 scripts which include:

1-preprocessing.py :
 this script includes preprocessing and visualizing English sentences and
it was given a text file that contains a number of paragraphs .
each paragraph in the script went through this pipeline:
- text -> preprocessed text -> 1-hot representation of words -> frequencies -> visualization .
 
2-similarty.py :
this script uses the BBC news data set which is a csv file that has two columns  one is the sentences and the other is the queries in order to trains and tests an NN model that finds the sentence most similar to a given query .

the pipeline of this script :
- query -> preprocess query -> compute query representation
- sentences -> preprocess sentences -> compute sentence representations
- select sentence such that similarity(query representation, sentence representation) is maximal
- train pipeline : training dataset -> entering sentences as labels and queries as data -> NN for regression ->Trained model
- test pipeline : 
give a new query after preprocessing to the trained dataset -> predict a sentence to the given query -> find the most similar sentence .
PS:
the words were encoded in two ways one is by using Word2vec and second is 1-hot encoding ,
.vec is needed to run this script and the BBC news dataset or something similar .

3-image_classification.py :

in this script we built an RNN model that takes as an input an image and produces label .
the dataset that is used in order to train this model is the 100-cifar data and the labels were produced by outputting a vector and then this vector was compared to other vectors in word2vec dataset and then choosing the top 3 most fitting words.

the script pipeline :
- Training: CIFAR-100 dataset and its labels -> CNN -> classification model
- Test: image ->preprocess image -> CNN -> get word vector of a label -> find 3 most relevant text labels  

in order to use this script a person needs the word2vec .vec file from google and the 100-cifar dataset.
