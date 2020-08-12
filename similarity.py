"""
This was done by :
Mohamed Abomokh 318245040
moataz atawna 207782319
Aladin handoklo 204161491‎

"""
""" ----- Libraries ----- """
import argparse
import io
import pandas as pd
import nltk as nl
from nltk.corpus import stopwords
from nltk.stem.porter import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.layers import Dropout
""" Args parser, In here we will parse the arguments passed in the terminal """
parser = argparse.ArgumentParser(
    description=" You can use this script to run train/test two models \n One is w2v and the other is N-hot encoded ",
    epilog="And that's it :) ... ")
parser.add_argument('--query', type=str, metavar='<query file global path>',
                    help="when using this option you have to enter a query text file path for testing")
parser.add_argument('--text', type=str, metavar='<text file global path>',
                    help="When using this option you need to enter a text file path that contains sentences "
                         "for testing")
parser.add_argument('--task', choices=['train', 'test'], type=str,
                    help="You have to choose whether you want to train/test one the models")
parser.add_argument('--data', type=str, metavar='<training Data set in .csv format>',
                    help="Here you enter that data set you wanna train")
parser.add_argument('--model', type=str, metavar='<trained model>',
                    help="When using this option you have to enter the trained Model file path")
parser.add_argument('--representation', choices=['w2v', 'n-hot'], type=str, metavar='<w2v | n-hot>',
                    help="When using this option you have to enter the representation you would want to work with")
args = parser.parse_args()

""" a vector for each word """


def load_vectors(fname):
    """

    :param fname: this is the file which from it we can load words with their vector embedded representation
    :return:it will return a dictionary where the the key is the word from the data set and the value is the vector
    """
    print("--- Loading vectors ---")
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))

    return data


"""
:parameter:w2v_dict is a global dictionary that the keys in it are words and values are a 300 shaped vector which is
the embedding of the word 
"""
if args.representation == 'w2v':
    w2v_dict = load_vectors('wiki_data.vec')


def prerocessing(sentences):
    """
    parameter: The column of either the tags or description
    return:An array of all the column as an array after cleaning
    """

    sentencearray = []
    # when it's n-hot then we want stem the words
    if args.representation == 'n-hot':
        stemmer = PorterStemmer()
    for sent in sentences:
        tokens = nl.word_tokenize(sent)
        Justwords = list(map(lambda x: " ".join(re.findall(r"\b[A-Za-z]+\b", x)), tokens))
        Justwords = ' '.join(Justwords).split()
        stop = stopwords.words("english")
        Justwords = list(x for x in Justwords if x not in stop)
        # when it's n-hot then we want stem the words
        if args.representation == 'n-hot':
            Justwords = list(stemmer.stem(x) for x in Justwords)
        sentencearray.append(Justwords)
    return sentencearray


def Querypreprocessing(query):
    """
    Preprocessing the query ,tokenizing ,cleaning and stemming if we are choosing test with N hot encoded
    :parameter query: is a query could be tags or whatever
    :returns:the query after preprocessing
    """
    global args

    tokens = nl.word_tokenize(query)
    Justwords = list(map(lambda x: " ".join(re.findall(r"\b[A-Za-z]+\b", x)), tokens))
    Justwords = ' '.join(Justwords).split()
    stop = stopwords.words("english")
    Justwords = list(x for x in Justwords if x not in stop)
    # when it's n-hot then we want stem the words
    if args.representation == 'n-hot':
        stemmer = PorterStemmer()
        Justwords = list(stemmer.stem(x) for x in Justwords)

    return Justwords


def Query2Vec(query):
    """
    :parameter query: when using the test project with the w2v represenition
    :returns:the query after the words have been replaced with their vector representation
    """
    global w2v_dict
    for i in range(len(query)):
        if query[i] in w2v_dict:
            query[i] = w2v_dict[query[i]]
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
    global w2v_dict

    for sent in column:
        for i in range(len(sent)):
            if sent[i] in w2v_dict:
                sent[i] = w2v_dict[sent[i]]
            else:
                sent[i] = np.zeros(300)
    return column


def making_dict(sentences, N):
    """
    This function builds a sorted index dictionary by checking which are the most frequent words
    in the sentences and and make their indexes from 0 - N-1 and write it to a csv file for later use
    :parameter sentences:  the description column or any list of sentences
    :parameter N: Most N frequent numbers that we want
    :returns: A sorted dictionary by frequency with each word having an index between 0 - N-1
    """
    dictfreqofwords = {}
    # Taking the words and counting the frequency of each word
    for sen in sentences:
        for w in sen:
            if w not in dictfreqofwords:
                dictfreqofwords[w] = 1
            else:
                dictfreqofwords[w] += 1
    # sorting the dictionary of indexes based on the frequency of the words
    dictfreqofwords = {k: v for k, v in sorted(dictfreqofwords.items(), key=lambda item: item[1], reverse=True)}

    # making two lists one for the indexes and one for the words to so it can be saved for to csv file
    # also a dictionary of sorted indexes to be returned in used
    count = 0
    list_of_indexes = []
    list_of_words = []
    sorted_dict_indexes = {}
    for key, value in dictfreqofwords.items():
        if count >= N:
            break
        else:
            sorted_dict_indexes[key] = count
            list_of_indexes.append(count)
            list_of_words.append(key)
            count += 1
    # saving to a csv file for later use in testing the model
    to_csv = {'word': list_of_words, 'index': list_of_indexes}
    df = pd.DataFrame(to_csv)
    df.to_csv("words_n_indexes.csv", index=False)

    return sorted_dict_indexes


def buildingNhot(dictindexes, sentences, N):
    """
    In this function we build the 1 hot coded representation in the size of N
    for each sentence we get
    :param dictindexes: a dictionary of N most frequent words with indexes from 0 - N-1
    :param sentences: sentences we want represent with 1 hot encoded
    :param N: The size of our 1 Hot encoded representation
    :returns : sentences , where each sentence has been replaced by the representation
    """

    veclist = [0] * N
    finallist = []
    for sen in sentences:
        for w in sen:
            if w in dictindexes:
                val = int(dictindexes[w])
                veclist[val] = 1
            else:
                pass
        finallist.append(np.asarray(veclist))
        veclist = [0] * N
    return np.asarray(finallist)


def fromSent2NHot(query, sentences, N):
    """
    in this function we change the query and each sentence to the 1 one encoded representation
    :param query: the query we want to encode
    :param sentences: each sentence we get from the text file we encode it
    :param N: this for knowing what is the size of the 1 hot encoded vector would look like
    :returns : returning the query and the sentences after representing them with 1 hot encoded vector
    """
    # we read the csv file to make a dictionary of indexes that we will use to encode
    dict_indexes = {}
    reader = csv.reader(open("words_n_indexes.csv", 'r'))
    next(reader)
    for word, index in reader:
        dict_indexes[word] = index

    # encoding the sentences
    sents = buildingNhot(dict_indexes, sentences, N)

    # encoding the query
    veclist = [0] * N
    queryarray = []
    for w in query:
        if w in dict_indexes:
            val = int(dict_indexes[w])
            veclist[val] = 1
        else:
            pass
    queryarray.append(np.asarray(veclist))

    return np.asarray(queryarray), sents


def ModelingW2V(df):
    """
    In this function we are building a model that finds the sentence most similar to given query
    the function builds the model and then writes saves the trained model
    :parameter df: is the data set we are trying train the model with
    """
    print("--- Preprocessing ---")
    queries = prerocessing(df['tags'])
    sentences = prerocessing(df['description'])

    # we will take the dictionary of word embeddings
    print("--- vectoring the words ---")
    queries = FromWord2Vec(queries)
    sentences = FromWord2Vec(sentences)

    print("--- calculating average of each sentence ---")

    X = AvgEmbedding(queries)  # data
    Y = AvgEmbedding(sentences)  # labels

    print("--- Building the w2v model ---")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=20)

    model = Sequential()
    model.add(Dense(300, input_dim=300, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(300, activation='linear'))
    model.compile(optimizer='adam', loss="mean_squared_error", metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=10)
    score, accuracy = model.evaluate(X_test, y_test, batch_size=10)

    print("Model Score = {:.2f}".format(score))
    print("Accuracy = {:.2f}".format(accuracy * 100))

    # after Evaluating we save the model and the weights into the .h5 file

    print("--- saving the model to h5 file ---")

    model.save("final_model_w2v.h5")

    print("--- Training is done ... ---")


def TestingW2VModel():
    query = open(args.query, "r").read()
    sentences = nl.sent_tokenize(open(args.text, "r").read())

    # preprocessing
    print("--- Preprocessing ---")
    x = Querypreprocessing(query)
    y = prerocessing(sentences)

    # vectoring the query and sentence
    print("--- vectoring the words ---")
    x = Query2Vec(x)
    y = FromWord2Vec(y)

    # average of each sentence
    print("--- calculating average of each sentence ---")
    x = Avg2Query(x)
    y = AvgEmbedding(y)

    # making a dictionary where the key is averaged vector of the sentences and the value is the sentence itself
    dict_sentences = {}
    count = 0
    for key in y:
        dict_sentences[str(key)] = sentences[count]
        count += 1
    # And now we load the model and predict the right sentence for the given query
    print("--- Loading the model ---")
    loaded_model = load_model(args.model)
    print("--- Predicting an outcome ---")
    pred_y = loaded_model.predict(x.reshape(1, 300))

    print("---Calculating the similarity between the prediction and the sentences ---")
    max_similar = np.array([[0]])
    most_sim_sent = y[0]
    for i in range(len(y)):
        similar = cosine_similarity(pred_y.reshape(1, -1), y[i].reshape(1, -1))
        if similar[0] >= max_similar[0]:
            max_similar = similar
            most_sim_sent = y[i]

    print("The similarity between the prediction and the most similar phrase is :{:.2f}% ".format(
        max_similar[0][0] * 100))

    print('--- writing the sentence to txt file ---')

    text_file = open("most_similar.txt", "w")
    text_file.write(dict_sentences[str(most_sim_sent)])
    text_file.close()

    print("--- Testing is done ... ---")


def ModelingNHot(df, N):
    """
    This function will build a model using the one hot encoding representation for each sentence in the data set .
    :param N: can be changed Based on the content of the data set or if we want bigger N hot representation of a sentence
    :output: a model file for n-hot representation that will be used in the testing phase
    """

    # tokenizing and stemming and cleaning the data
    print("--- Preprocessing ---")
    tags = prerocessing(df['tags'])
    description = prerocessing(df['description'])

    # making a dictionary from the description
    print("--- making the one hot encoded vector ---")
    dict_of_indexes = making_dict(description, N)

    print("--- encoding the words ---")
    # building N hot representation for each sentence in the in the data set
    x = buildingNhot(dict_of_indexes, tags, N)  # data
    y = buildingNhot(dict_of_indexes, description, N)  # labels

    # building the model
    print("--- Building the n-hot model ---")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=20)

    model = Sequential()
    model.add(Dense(int(N / 20), input_dim=N, activation='sigmoid'))
    model.add(Dense(int(N / 10), activation='sigmoid'))
    model.add(Dense(N, activation='sigmoid'))
    model.add(Dense(N, activation='sigmoid'))
    model.add(Dense(N, activation='relu'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=200)
    score, accuracy = model.evaluate(x_test, y_test)

    print("Model Score = {:.2f}".format(score))
    print("Accuracy = {:.2f}".format(accuracy * 100))

    # after Evaluating we save the model and the weights into the .h5 file
    print("--- saving the model to h5 file ---")
    model.save("final_model_n-hot.h5")
    print("--- Done Training ... ---")


def TestingNHotModel(N):
    query = open(args.query, "r").read()
    sentences = nl.sent_tokenize(open(args.text, "r").read())

    # preprocessing
    print("--- Preprocessing ---")
    x = Querypreprocessing(query)  # data
    y = prerocessing(sentences)  # labels

    # representing each sentence and the query with on hot
    print("--- encoding the the sentences and the query ---")
    x, y = fromSent2NHot(x, y, N)

    # making a dictionary where the key is averaged vector of the sentences and the value is the sentence itself
    dict_sentences = {}
    count = 0
    for key in y:
        dict_sentences[str(key)] = sentences[count]
        count += 1

    # And now we load the model and predict the right sentence for the given query
    print("--- Loading the model ---")
    loaded_model = load_model(args.model)
    print("--- Predicting an outcome ---")
    pred_y = loaded_model.predict(x)

    # making the predction to 1 hot where the number is going to be turned to 1
    # if it's bigger than than the avg number in the vector else 0
    arg_max = np.argmax(pred_y[0], axis=0)
    arg_max = pred_y[0][arg_max]
    arg_min = np.argmin(pred_y[0], axis=0)
    arg_min = pred_y[0][arg_min]
    arg_avg = (abs(arg_max) + abs(arg_min)) / 2
    for i in range(len(pred_y[0])):
        if pred_y[0][i] > arg_avg / 2:
            pred_y[0][i] = 1
        else:
            pred_y[0][i] = 0

    # checking for most similar sentence by using cosine similarity
    print("---Calculating the similarity between the prediction and the sentences ---")
    max_similar = np.array([[0]])
    most_sim_sent = y[0]
    for i in range(len(y)):
        similar = cosine_similarity(pred_y.reshape(1, -1), y[i].reshape(1, -1))
        if similar[0] >= max_similar[0]:
            max_similar = similar
            most_sim_sent = y[i]

    print("The similarity between the prediction and the most similar phrase is :{:.2f}% ".format(
        max_similar[0][0] * 100))
    print('--- writing the sentence to txt file ---')

    text_file = open("most_similar.txt", "w")
    text_file.write(dict_sentences[str(most_sim_sent)])
    text_file.close()

    print("--- Testing is done ... ---")


if __name__ == "__main__":
    if args.task == 'train':
        df = pd.read_csv(args.data)
        df = df.dropna()
        if args.representation == 'w2v':
            ModelingW2V(df)
        elif args.representation == 'n-hot':
            ModelingNHot(df, 10000)

    if args.task == 'test':
        if args.representation == 'w2v':
            TestingW2VModel()
        elif args.representation == 'n-hot':
            TestingNHotModel(10000)
