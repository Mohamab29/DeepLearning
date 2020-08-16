""" ----- Libraries ----- """

from keras.layers import BatchNormalization
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
import argparse
import io
import nltk as nl
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from time import sleep
from time import time
from nltk.stem.porter import *
import cv2
from nltk.corpus import stopwords

"""###########################################

This was done by :
Mohamed Abomokh 
moataz atawna 
Aladin handoklo 

###########################################"""

""" ---- argument parsing ---- """

""" Args parser, In here we will parse the arguments passed in the terminal """
parser = argparse.ArgumentParser(description="To run the script for tain/test")
parser.add_argument('--image', type=str, metavar='<image file global path>',
                    help='you need to enter an image file path for testing')
parser.add_argument('--task', choices=['train', 'test'], type=str, help="Choose between train/test the models")
parser.add_argument('--model3', type=str, metavar='<CNN model file name>', help="Enter trained model 3 file path ")
parser.add_argument('--model4', type=str, metavar='<RNN model file name>', help="Enter trained model 4 file path ")
parser.add_argument('--model2', type=str, metavar='<Regression model that predicts sentence given keywords>',
                    help="Enter trained model 2 file path ")
parser.add_argument('--wordvec', type=str, metavar='<word vectors file>',
                    help='wordvec: path of FastText wv file (*.vec)')
args = parser.parse_args()


def calculate_time(start_time):
    """
    calculates the time from start to current time and returns it in secedes
    :param start_time: this is the time from which we want to calculate how much time has passed since this time
    :returns:the current time
    """
    return round(time() - start_time, 2)


""" We load the vectors and make a dictionary from the .vec file"""


def load_vectors(fname):
    """

    :param fname: this is the file which from it we can load words with their vector embedded representation
    :return:it will return a dictionary where the the key is the word from the data set and the value is the vector
    """
    sTime = time()
    print("--- Loading vectors ---")
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    m, s = divmod(calculate_time(sTime), 60)
    print(f"--- Done Loading vectors in {int(m):02d}:{int(s):02d} minutes ---")
    sleep(0.5)
    return data


"""
   Global dictionary that we will use throughout the script ,
   change the path or name of the file based 
   on how it's called at your directory
"""
if args.wordvec:
    dict_words_n_vectors = load_vectors(args.wordvec)
else:
    print("You didn't enter the word2vec global path.")


def Query2Vec(query):
    """
    :parameter query: when using the test project with the w2v represenition
    :returns:the query after the words have been replaced with their vector representation
    """
    global dict_words_n_vectors
    for i in range(len(query)):
        if query[i] in dict_words_n_vectors:
            query[i] = dict_words_n_vectors[query[i]]
        else:
            query[i] = np.zeros(300)
    return query


def Avg2Query(query):
    """
    :parameter:the query we would like to calculate the mean of the vector
    :returns:the avg of a query vector
    """
    return np.asarray(np.mean(query, axis=0))


def AvgEmbedding(column):
    """
    :param column: each sentence in the column that enters well have it's avg word embedding calculated
    :return: the column avg embedding calculated for each sentence
    """
    Avgembed = []
    for i in column:
        Avgembed.append(np.mean(i, axis=0))
    return np.asarray(Avgembed)


def FromWord2Vec(column):
    """
    :param column: an array of column tags/description
    :return: words in each array replaced with it's vector embedded reprehension
    """
    global dict_words_n_vectors

    for sent in column:
        for i in range(len(sent)):
            if sent[i] in dict_words_n_vectors:
                sent[i] = dict_words_n_vectors[sent[i]]
            else:
                sent[i] = np.zeros(300)
    return column


def preprocessing(sentences):
    """
    parameter: The column of either the tags or description
    return:An array of all the column as an array after cleaning
    """
    sentencearray = []

    for sent in sentences:
        tokens = nl.word_tokenize(sent)
        Justwords = list(map(lambda x: " ".join(re.findall(r"\b[A-Za-z]+\b", x)), tokens))
        Justwords = ' '.join(Justwords).split()
        if args.task == "test":
            stop = stopwords.words("english")
            Justwords = list(x for x in Justwords if x not in stop)
        sentencearray.append(Justwords)
    return sentencearray


def seq2Vec(sequences):
    """
    in this function we convert the each word in each sequence to word embedded  word using word2vec
    :param sequences: our sequences that we want to convert
    :returns sequences: after each word has been replaced
    """
    global dict_words_n_vectors
    for sent in sequences:
        for i in range(len(sent)):
            if sent[i] in dict_words_n_vectors:
                sent[i] = dict_words_n_vectors[sent[i]]
            else:
                sent[i] = np.zeros(300)
    return np.array(sequences, dtype="float32")


def labels2Vec(labels):
    """
    in this function we convert the each word in each label to word embedded  word using word2vec
    :param labels: our labels that we want to convert
    :returns labels: after each word has been replaced
    """
    global dict_words_n_vectors

    for i in range(len(labels)):
        if labels[i] in dict_words_n_vectors:
            labels[i] = dict_words_n_vectors[labels[i]]
        else:
            labels[i] = np.zeros(300)
    return np.array(labels, dtype="float32")


def sents_to_seqs(sentences):
    """
    In this function we are making a sequences and labels for each sentence and for all the data set with same data type
    :param sentences: our sentences that we want to convert from a sentence to list of sequences and labels
    :returns:list of sequences and list of labels
    ps:
    we choose to work with a sequence of length 4 cause we thought it was the best choice and could give the best
    results .
    """
    list_of_sent = []
    list_of_seq = []
    list_of_labels = []
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            if j + 4 < len(sentences[i]):
                list_of_sent.append(sentences[i][j])
                list_of_sent.append(sentences[i][j + 1])
                list_of_sent.append(sentences[i][j + 2])
                list_of_sent.append(sentences[i][j + 3])
                label = sentences[i][j + 4]
                list_of_labels.append(label)
                list_of_seq.append(list_of_sent)
                list_of_sent = []

    return list_of_seq, list_of_labels


def resize_image():
    """this function resize an image to the form of (32,32,3) so we can use it to predict a label
        :returns: an array representation of an image
    """
    print("--- resizing the image ---")
    if args.image:
        img = cv2.imread(args.image)
        img = cv2.resize(img, (32, 32))
        img = np.array(img, dtype=np.uint8)
        img = img.reshape(32, 32, 3) / 255
        img = np.expand_dims(img, axis=0)
        print("--- done resizing the image ---")
        return img
    else:
        print("Please enter a path of an image to check for it's label")


def most_similar(prediction):
    """
    in this function we will check which of the vectors in the word vector dictionary
    is most similar to our prediction and return the word and it's vector for later use
    :param prediction: the vector that we want to calculate it's cosine similarity with each
    vector in the w2v dictionary
    :returns : word that is most similar and vector of embedded representation
    """
    sTime = time()
    max_prediction = np.array([[0]])
    for key, value in dict_words_n_vectors.items():
        sim = cosine_similarity(prediction.reshape(1, -1), value.reshape(1, -1))
        if sim[0] > max_prediction[0]:
            max_prediction = sim
            word, vector = key, value
    m, s = divmod(calculate_time(sTime), 60)
    print(f"--- done checking most similar word in {int(m):02d}:{int(s):02d} minutes --- ")
    return word, np.expand_dims(np.asarray(vector), axis=0)


def expanding_shape(prediction):
    """
    In this helper function we expand the dimension of a given prediction to 3 dims and shape (4,len(prediction))
    :param prediction: the prediction that we want to expand
    :return: expanded prediction
    """
    expanded_prediction = [
        prediction[0],
        prediction[0],
        prediction[0],
        prediction[0]
    ]
    return np.expand_dims(np.asarray(expanded_prediction), axis=0)


# train the model
def train_rnn(df):
    """
    In this function we will train our RNN model using the BBC data set news
    :type df: a column from in a data frame (pandas data frame)
    :param df: our data set that has to have column called description which contains a column of sentences
    """
    print("--- training RNN(LSTM) model ---")
    sTime = time()
    print("--- preprocessing the data ---")
    sentences = preprocessing(df['description'])

    print("--- making sequences and labels ---")
    seq, labels = sents_to_seqs(sentences)

    print("--- Converting words in sequences labels into vectors ---")
    seq = seq2Vec(seq)
    labels = labels2Vec(labels)
    # splitting the data to 80/20
    x_train, x_test, y_train, y_test = train_test_split(seq, labels, test_size=0.2, shuffle=42)

    # memory cleaning
    seq = []
    labels = []
    sentences = []

    # building our RNN model
    print("--- building our RNN model ---")
    model = Sequential()

    model.add(LSTM(300, return_sequences=True, activation="tanh", input_shape=(4, 300)))
    model.add(Dropout(0.3))
    model.add(LSTM(300, return_sequences=True, activation="tanh"))
    model.add(Dropout(0.3))
    model.add(LSTM(300, return_sequences=True, activation="tanh"))
    model.add(Dropout(0.3))
    model.add(LSTM(300, return_sequences=False, activation="tanh"))
    model.add(Dense(300, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(250, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(250, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(300, activation="linear"))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    print("--- fitting the data to our model ---")

    model.fit(x_train, y_train, epochs=10, batch_size=30000)

    x_train = []
    y_train = []

    score, accuracy = model.evaluate(x_test, y_test, batch_size=4000)

    print("Model Score = {:.2f}".format(score))
    print("Accuracy = {:.2f}".format(accuracy * 100))
    model.save("text_model.h5")
    print("--- text_model has been saved to current working directory ---")

    m, s = divmod(calculate_time(sTime), 60)
    print(f"--- Done training our RNN model in {int(m):02d}:{int(s):02d} minutes ---")


def testing_regression_cnn_rnn(df):
    """
    In this function we will test our rnn model using an image and a CNN model
    our CNN model will predict a certain word for the given image in the terminal and produce a word that fits the word
    after that we enter this word to our RNN model and then again we enter the word that comes our to the rnn model to get
    a second word and eventually will have a headline for an image of size 3
    And if we are running the regression model then we will use the sentences from the data set in order to see which
    sentence is the most similar for the headline that we got from running model 3 and 4 .
    :param df: our sentences that we will use to in our regression model
    """

    if args.model4 and args.model3:
        sTime = time()
        print("--- First we load the trained the models ---")
        rnn_model = load_model(args.model4)
        cnn_model = load_model(args.model3)

        # now we load and resize our image from the path that we got in the command line
        image = resize_image()

        # making a prediction for the image
        print("--- predicting first word ---")
        cnn_prediction = cnn_model.predict(image)
        # getting most similar label for the image and it's vector word embedded representation
        f_word, f_vector = most_similar(cnn_prediction)

        print("--- predicting second word ---")
        # getting second prediction
        rnn_first_prediction = rnn_model.predict(expanding_shape(f_vector))
        # second word and vector
        s_word, s_vector = most_similar(rnn_first_prediction)

        print("--- predicting third word ---")
        # and finally
        rnn_second_prediction = rnn_model.predict(expanding_shape(s_vector))
        t_word, t_vector = most_similar(rnn_second_prediction)

        print("--- This is the our predicted headline for the given image:\n")
        print(f_word, s_word, t_word)

        m, s = divmod(calculate_time(sTime), 60)
        print(f"\n--- done testing in model 3 and 4 in {int(m):02d}:{int(s):02d} minutes --- ")
        if args.model2:
            print("--- Now we are loading the regression model to find the sentence that's similar the headline ---")
            regression_model = load_model(args.model2)
            query = [f_word, s_word, t_word]
            description = df['description']
            sentences = []
            for sent in description:
                sentences.append(sent)

            # preprocessing
            print("--- preprocessing the data ---")
            x = query
            y = preprocessing(description)

            # vectoring the query and sentence
            print("--- vectoring the data ---")
            x = Query2Vec(x)
            y = FromWord2Vec(y)

            # average of each sentence
            print("--- calculating the averages ---")
            x = Avg2Query(x)
            y = AvgEmbedding(y)

            # making a dictionary where the key is averaged vector of the sentences and the value is the sentence itself
            dict_sentences = {}
            for i, key in enumerate(y):
                dict_sentences[str(key)] = str(sentences[i])

            # And now we load the model and predict the right sentence for the given query
            print("--- predicting a sentence ---")
            pred_y = regression_model.predict(x.reshape(1, 300))
            print("--- finding most similar sentence ---")
            max_similar = np.array([[0]])
            most_sim_sent = y[0]
            for i in range(len(y)):
                similar = cosine_similarity(pred_y.reshape(1, -1), y[i].reshape(1, -1))
                if similar[0] >= max_similar[0]:
                    max_similar = similar
                    most_sim_sent = y[i]

            print("--- The most similar sentence is:\n")
            print(dict_sentences[str(most_sim_sent)])

            m, s = divmod(calculate_time(sTime), 60)
            print(f"\n--- done testing all models in {int(m):02d}:{int(s):02d} minutes --- ")
    else:
        print("Please train or enter the models file path in order to test")


if __name__ == "__main__":
    """
    the BBC news is hard coded , if you have other data 
    sets or different file name please change it here .
    """
    df = pd.read_csv("BBC_dataset.csv")
    df = df.dropna()
    if args.task == "train":
        train_rnn(df)
    elif args.task == "test":
        testing_regression_cnn_rnn(df)
    else:
        print("You didn't enter a task type right\nplease enter train or test.")
