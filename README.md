#DeepLearning

in the course of deep learning we learned the basics of deep learning and learned different algorithms  Which are CNN ,  RNN , LSTM , NN for find the sentence most similar to given query  and more
In the four files You see in this repository there are 4 scripts and a text file which include :

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

4-image_headline.py

in this script we built an LSTM model that generates a headline for a given image
we train or use trained models from the pervious two scripts mentioned above (similarity and image classification) in order to predict and headline for a given image first by outputting a query that consists of three words which we then feed to the second model the finds the most similar sentence to that query .

the pipeline in the script :
- Training pipeline:
trains two networks:
N1 - image classification network that produces a word describing an image from lab 3.
N2 - an LSTM network that predicts the next word given a word.

- Test pipeline: 
image -> image headline
we use N1 to produce a word W1 for given image, then we use N1 to produce two more words W2,W3 to build a headline.
W2 is prediction by N2 given W1, and W3 is prediction by N2 given W2.
N4 produces word vectors, in order to print words, we find which word it is by using WV dictionary (word2vec) and cosine similarity.
and then use network N3 from similarty.py to find the sentence that matches our 3 keywords from N2, and print it as an alternative headline.

In order to run this script we need these datasets :
the word2vec .vec file from google and the 100-cifar dataset and the bbc news dataset.

PS:

In order to run scripts from 2 to 4 you need to run them in the terminal in the same working dirctory with all the datasets there and enter the arguments found in the text file
