#DeepLearning
in the course of deep learning we learned the basics of deep learning and learned different algorithms  Which are CNN ,  RNN , LSTM , NN for find the sentence most similar to given query  and more
In the four files You see in this repository there are 4 scripts which include:
1- preprocessing.py :
 this script includes preprocessing and visualizing English sentences and
it was given a text file that contains a number of paragraphs .
each paragraph in the script went through this pipeline:
 text -> preprocessed text -> 1-hot representation of words -> frequencies -> visualization .
2- similarty.py :
this script uses the BBC news data set which is a csv file that has two columns  one is the sentences and the other is the queries in order to trains and tests an NN model that finds the sentence most similar to a given query .
the pipeline of this script :
query -> preprocess query -> compute query representation
sentences -> preprocess sentences -> compute sentence representations
select sentence such that similarity(query representation, sentence representation) is maximal
- train pipeline : training dataset -> entering sentences as labels and queries as data -> NN for regression ->Trained model
- test pipeline : 
give a new query after preprocessing to the trained dataset -> predict a sentence to the given query -> find the most similar sentence .
PS:
the words were encoded in two ways one is by using Word2vec and second is 1-hot encoding ,
.vec is needed to run this script and the BBC news dataset or something similar .
