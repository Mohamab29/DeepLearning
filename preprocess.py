"""
This was done by :
Mohamed Abomkh 318245040
moataz atawna 207782319
Aladin handoklo 204161491‎


"""
import argparse
import nltk as nl
from nltk.corpus import stopwords
import wordcloud as wc
from nltk.stem.porter import *
from itertools import islice
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

"""First we open the text file and take the data using argsparse"""
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('txtfile', metavar='Text file', type=str, nargs='?',
                    help='an integer for the accumulator',default="sample.txt")
args = parser.parse_args()
textfile = args.txtfile
t = open(textfile, encoding="utf8", errors='ignore').read()

"""tokenizing the words"""
word_tokens = nl.word_tokenize(t)
sent_tokens = nl.sent_tokenize(t)
Listofsents = []
ListofWords = []
# for stemming
stemmer = PorterStemmer()

"""After we've tokenized we now have to remove stopwords and . / from each sentence """
for sent in sent_tokens:
    tokens = nl.word_tokenize(sent)
    Justwords = list(map(lambda x: " ".join(re.findall(r"\b[A-Za-z]+\b", x)), tokens))
    Justwords = ' '.join(Justwords).split()
    stop = stopwords.words("english")
    Justwords = list(x for x in Justwords if x not in stop)
    Justwords = list(stemmer.stem(x) for x in Justwords)
    Listofsents.append(Justwords)
    ListofWords += Justwords

"""Word cloud"""
wordswiththeirFreqs = dict()
# counting the frequencies of the words
for word in ListofWords:
    word = word.lower()
    wordswiththeirFreqs[word] = wordswiththeirFreqs.get(word, 0) + 1

word_cloud = wc.WordCloud(
    background_color="black",
    width=500,
    height=500)


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


max_sorted_Freqs = {k: v for k, v in sorted(wordswiththeirFreqs.items(), key=lambda item: item[1], reverse=True)}
# we take the top 20 words
word_cloud.generate_from_frequencies(dict(take(20, max_sorted_Freqs.items())))
word_cloud.to_file(args.txtfile.replace('txt','_cloud.png'))

"""encoding the words """


def Hotencoding(word_tokens, sentences):
    """a hot encoding function that outputs a text file of sentences with vectors of binary encoded labels in
     a matrix with each row being a sentence and columns as the vectors """

    # first we make arrays from the values
    values = array(word_tokens)
    label_encoder = LabelEncoder()

    # here we make labels for each word
    # from 0-402 labels in integers
    integer_encoded = label_encoder.fit_transform(values)

    # and here we make the binary encoding by using the one hot encoder which is function that has the fit transforms
    # with using the integers we had build from label encoder it turns each label to it's binary form for example
    # the word the label 4 will have a binary encode of [0,0,0,0,1,...,0]
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # and now we make a dictionary of the the encoded values with each of the words
    Dict_of_encoded_words = {}
    counter = 0
    for i in values:
        Dict_of_encoded_words[i] = list(onehot_encoded[counter])
        counter += 1

    # and now that we have the dict of words with there encoded values we
    # can now make a matrix of sentences and there binary codes

    Encoded_sentences = sentences.copy()

    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            Encoded_sentences[i][j] = Dict_of_encoded_words[sentences[i][j]]
    # after that we putted them in a list containg a matrix of sentences with their words being encoded
    # we output the results in a text file

    with open(args.txtfile.replace('txt','_1hot.txt'), 'w') as f:
        for item in Encoded_sentences:
            f.write("%s\n" % item)


"""To start to hot coding"""
Hotencoding(ListofWords, Listofsents)
