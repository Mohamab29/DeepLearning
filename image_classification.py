""" ----- Libraries ----- """
import pickle
import io
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import operator
import cv2
import numpy as np
import argparse
from time import sleep
from time import time

"""###########################################"""
"""
This was done by :
Mohamed Abomokh 318245040
moataz atawna 207782319
Aladin handoklo 204161491‎
"""
"""###########################################"""

""" ---- argument parsing ---- """

""" Args parser, In here we will parse the arguments passed in the terminal """
parser = argparse.ArgumentParser(
    description=" You can use this script to run train/test a CNN model while using CIFAR-100 data ",
    epilog="And that's it :) ... ")

parser.add_argument('--task', choices=['train', 'test'], type=str,
                    help="You have to choose whether you want to train a model or to test it ")
parser.add_argument('--image', type=str, metavar='<image file name>',
                    help="Here you enter the image file path")
parser.add_argument('--model', type=str, metavar='<trained model>',
                    help="When using this option you have to enter the trained Model file path")
args = parser.parse_args()


def calculate_time(Starttime):
    """
    calculates the time from start to current time and returns it in secedes
    :param Starttime: this is the time from which we want to calculate how much time has passed since this time
    :returns:the current time
    """
    return round(time() - Starttime, 2)


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

dict_words_n_vectors = load_vectors('wiki_data.vec')


def unpickle(file):
    """
    unpickling the meta,train,test CIFAR files
    :returns: returning a dict from the pickled files
    """

    with open(file, 'rb') as fo:
        _dict = pickle.load(fo, encoding='bytes')
    return _dict


def vectorized_labels(fine_labels):
    """
    this functions turns the data set labels to their vector representation from the w2v data set
    :param fine_labels:the labels we want to turn and in our models case their are 100 labels 
    :returns: a list of the words as vectors of size 100 x 300
    """
    global dict_words_n_vectors
    labels = []
    print("--- turning labels to word vectors ---")
    for label_name in fine_labels:
        if '_' not in label_name:
            labels.append(dict_words_n_vectors[label_name])
        else:
            two_words = label_name.split('_')
            word1_vector = dict_words_n_vectors[two_words[0]]
            word2_vector = dict_words_n_vectors[two_words[1]]
            two_words_avg = (word1_vector + word2_vector) / 2
            labels.append(two_words_avg)
    print("--- Done turning labels into vectors ---")
    sleep(0.5)
    return labels


def image_shaping(data):
    """
    :param data:is a list of images where each element is in size 3072
    :returns: an array where each element is an image shaped (32,32,3)
    """
    images = list()
    """Reshaping each image to to the shape (32,32,3)"""
    print("--- reshaping the data of an img to the right shape for train data---")
    for d in data:
        image0 = np.zeros((32, 32, 3), dtype=np.uint8)
        image0[..., 0] = np.reshape(d[:1024], (32, 32))  # Red channel
        image0[..., 1] = np.reshape(d[1024:2048], (32, 32))  # Green channel
        image0[..., 2] = np.reshape(d[2048:], (32, 32))  # Blue channel
        images.append(image0 / 255)
    print("--- Done reshaping the image ---")
    return np.asarray(images)


def from_labels_2vectors(labels, dataset_labels):
    """
    :param dataset_labels: our data set labels that are as vectors and we want to use to turn our labels into vectors
    :param labels: the labels of the test or train images that we want to make as vectors
    :returns: an array containing vectorized label for each image
    """

    vectorized_label_s = []
    print("--- Vectoring the labels of the test/train labels --- ")
    for i in range(len(labels)):
        # each labels[i] is an index for the list of the dataset_labels ,ex : label[0]=23 => dataset[23] => our label
        index = labels[i]
        label = dataset_labels[index]
        vectorized_label_s.append(label)

    print("--- Done turning test/train labels into vectors  ---")
    sleep(0.5)
    return np.asarray(vectorized_label_s)


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


def check_most_similar_labels(prediction):
    """
    in this function we will check which of the vectors in the word vector dictionary
    is most similar to our prediction and print the top 3 most similar words
    :param prediction: our label that we wanna see which vector is most similar to it
    :returns : after checking which vectors are most similar and sorting them from biggest to smallest
    we return a list of top 3 words that their vector was the most similar
    """
    global dict_words_n_vectors
    sTime = time()
    print("--- checking the most similar words to our predicted label ---")
    dict_result = {}
    i = 0
    for key, value in dict_words_n_vectors.items():
        similar = cosine_similarity(prediction.reshape(1, -1), value.reshape(1, -1))
        dict_result[key] = similar[0][0] * 100

    sorted_d = dict(sorted(dict_result.items(), key=operator.itemgetter(1), reverse=True))
    Top_similar = []
    for key, value in sorted_d.items():
        if i <= 2:
            Top_similar.append((key, value))
        else:
            break
        i += 1

    m, s = divmod(calculate_time(sTime), 60)
    print(f" --- done checking most similar words in {int(m):02d}:{int(s):02d} minutes --- ")

    return Top_similar


def train_the_model():
    """In this function we train our CNN with fully connected NN ,
     in the end the function saves the model after training """

    # First we unpickle the meta file to get the fine labels from it and also turning them to vectors afterwords
    # meta , test and train files that we unpickle are found in the CIFAR-100 data set website
    meta_file = unpickle('meta')
    dataset_labels = vectorized_labels([t.decode('utf8') for t in meta_file[b'fine_label_names']])

    # Unpickling the the test data
    train = unpickle('train')
    test = unpickle('test')

    # Taking the labels for each image in the test and data
    train_fine_labels = train[b'fine_labels']
    test_fine_labels = test[b'fine_labels']

    # our images
    train_data = train[b'data']
    test_data = test[b'data']

    # making our x and y train
    x_train = image_shaping(train_data)
    y_train = from_labels_2vectors(train_fine_labels, dataset_labels)
    # first we get our x and y test data and labels
    x_test = image_shaping(test_data)
    y_test = from_labels_2vectors(test_fine_labels, dataset_labels)

    # clearing memory
    train = []
    test = []
    train_data = []
    test_data = []
    train_fine_labels = []
    test_fine_labels = []
    dataset_labels = []

    # building our model
    print("--- building the model ---")
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(Dense(300, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(Dense(100,activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("--- fitting the data to our model ---")
    sTime = time()
    model.fit(x_train, y_train, epochs=25, batch_size=1000)

    m, s = divmod(calculate_time(sTime), 60)
    print(f"--- Done training our model in {int(m):02d}:{int(s):02d} minutes ---")
    sleep(1)

    print("--- evaluating the model and then saving it ---")
    sTime = time()

    # evaluate
    score, accuracy = model.evaluate(x_test, y_test, batch_size=500)
    # the results
    print("Our Models' Score = {:.2f}".format(score))
    print("The Accuracy of the model = {:.2f}".format(accuracy * 100))

    # after Evaluating we save the model and the weights into the .h5 file
    print("--- saving the model ---")
    model.save("image_model.h5")
    m, s = divmod(calculate_time(sTime), 60)
    print(f" --- done evaluating and saving in {int(m):02d}:{int(s):02d} minutes --- ")


def test_the_model():
    """in this function we will test the model that we have built and trained , if the .h5 is not found
        that means you need to first build the model before testing
    """
    if args.model:
        sTime = time()
        print("--- First we load the trained the model ---")
        loaded_model = load_model(args.model)

        # now we load and resize our image from the path that we got in the command line
        image = resize_image()

        # making a prediction
        prediction = loaded_model.predict(image)

        m, s = divmod(calculate_time(sTime), 60)
        print(f"--- done loading the model and predicting a label in {int(m):02d}:{int(s):02d} minutes --- ")
        top_similar = check_most_similar_labels(prediction)
        print("These are the top similar labels to our prediction :")
        for word, similarity in top_similar:
            print(" {0} ".format(word), end="")

        m, s = divmod(calculate_time(sTime), 60)
        print(f"\n--- done testing in {int(m):02d}:{int(s):02d} minutes --- ")
    else:
        print("Please train or enter the model file path in order to test")


if __name__ == "__main__":
    if args.task:
        if args.task == "train":
            train_the_model()

        if args.task == 'test':
            test_the_model()
    else:
        print("Please enter a task to be able to run the script")
