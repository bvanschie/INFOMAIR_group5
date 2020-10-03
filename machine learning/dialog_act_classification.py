from typing import List, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
from torch.nn import Module, Linear, ReLU, Sequential, Sigmoid, Dropout
from torch.nn.functional import cross_entropy
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import itertools
import json
import pickle
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tensorflow.keras.models import Sequential, save_model, load_model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

TRAIN_NN_TF = True
TRAIN_NN_PYTORCH = False

################################################
############## Utility Functions ###############
################################################

def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the data file from the provided path.
    Returns a tuple of numpy arrays, each with shape (N,) where N is
    the batch size. The first element contains a list of utterances, the
    second element contains the corresponding labels (as string).
    """
    with open(path, 'r') as file:
        lines = file.readlines()
    data = list(map(lambda s: [s.split()[0].lower(), ' '.join(s.split()[1:]).lower()], lines))
    data = np.array(data).T
    return data[1], data[0]

def preprocess_data(text: np.ndarray, vectorizer: CountVectorizer) -> np.ndarray:
    """
    Converts an array of strings into their bag of words representation
    using the provided vectorizer.
    :param text: np array of strings of shape (B,)
    :param vectorizer: The vectorizer to use.
    :return: A numpy array of shape (B, W) where B is the batch size (number of input strings)
    and W is the number of words used by the vectorizer in the bag of words prepresentation.
    """
    out = vectorizer.transform(text)
    out = np.array(out.toarray())
    return out

def preprocess_labels(labels: np.ndarray, mapping: np.ndarray):
    """
    Converts an array labels (string) into a numpy array of the
    corresponding class indices.
    :param labels: The list of string labels.
    :param mapping: An array containing all the labels. This list will
    be used as basis to determining the labels index.
    Eg.: if mapping = ['label1', 'label2', 'label3'] then
    all entries of 'labels' containing label1 will be mapped to 0,
    label2 to 1, and label3 to 2.
    :return: A numpy array with the index of labels.
    """
    out = [mapping.tolist().index(l) for l in labels]
    return np.asarray(out)

class MajorityClassifier:
    """
    This classifier always returns the label for the majority class.
    """
    def __init__(self, training_labels: list):
        """
        Init.
        :param return_value: The value to return. Should be the
        index of the majority class in the dataset.
        """
        c = Counter(labels);
        most_common_label = c.most_common(1)[0][0]
        self.return_value = list(LABELS).index(most_common_label)

    def __call__(self, texts: np.ndarray) -> np.ndarray:
        return np.full(shape=texts.shape, fill_value=self.return_value)

class KeywordClassifier:
    """
    This classifier uses a keyword list to determine the
    dialog act of an utterance.
    """
    def __init__(self, keyword_dict: str):
        """
        Init.
        :param keyword_dict: Path to the file containing the keywords for
        the different labels.
        """
        with open(keyword_dict, 'r') as file:
            self.keywords = json.load(file)
        self.label_names = LABELS.tolist()

    def __call__(self, texts: np.ndarray) -> np.ndarray:
        def find_keyword(s):
            """
            Finds the first label such that the text contains a word
            in the list of the label's keywords.
            :param s: The text to find the keyword in.
            :return: The first matching label.
            """
            for k, _ in self.keywords.items():
                if k in s.split():
                    return k

            # If no keyword is matched return majority class.
            return 'inform'

        predictions = [find_keyword(s) for s in texts]
        return np.array([self.label_names.index(p) for p in predictions])

def test_classifier(classifier, data: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Evaluates the classifier on the given batch. Outputs the accuracy and the confusion matrix.
    :param classifier: The classifier to evaluate.
    :param data: The machine training data.
    :param labels: Test labels.
    :return: A tuple of (accuracy, confusion matrix).
    """
    # If the classifier is a torch module the input type should be a tensor
    # and not a numpy array.
    if isinstance(classifier, Module):
        data = torch.tensor(data, dtype=torch.float32)

    predictions = classifier(data)
    golden_labels = preprocess_labels(labels, mapping=LABELS)
    acc = accuracy_score(y_true=golden_labels, y_pred=predictions)
    conf_mat = confusion_matrix(y_true=golden_labels, y_pred=predictions)

    return acc, conf_mat

def plot_training_history(history):
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()



################################################
############## Data preprocessing ##############
################################################



data, labels = load_data("../data/dialog_acts.dat")
SHOW_STATS = False
LABELS = np.unique(labels)
print(f"The labels in this dataset are: {LABELS}")

# The ratio of training data.
split = .85

split = int(split * data.shape[0])
X_train = data[:split]
y_train = labels[:split]
X_test = data[split:]
y_test = labels[split:]

vectorizer_train = CountVectorizer()

bag_of_words = vectorizer_train.fit_transform(X_train)
tfidf = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf.fit_transform(bag_of_words)


if SHOW_STATS:
    # number of words
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_sents = [tokenizer.tokenize(s) for s in X_train]
    words = list(itertools.chain.from_iterable(tokenized_sents))
    print(f"Total amount of words: {len(words)}")

    # Vocabulary
    vocabulary = list(vectorizer_train.vocabulary_)
    print(f'Vocabulary size: {len(vocabulary)}')

    # Out of vocabulary words
    out_of_words = []
    vectorizer_test = CountVectorizer()
    bag_of_words_test = vectorizer_test.fit_transform(X_test)
    vocabulary_test = list(vectorizer_test.vocabulary_)

    for word in vocabulary_test:
        if word not in vocabulary:
            out_of_words.append(word)

    print(f'Number of out-of-vocabulary-words: {len(out_of_words)}')

    # Make histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.random.normal(0, 1, 1000)
    numBins = 15

    ax.hist(preprocess_labels(labels, mapping=LABELS), bins=np.arange(numBins + 1) - 0.5, color='#1f77b4')
    ax.set_xticklabels(LABELS, rotation=55)

    plt.xlabel('Dialog act')
    plt.ylabel('Count')

    plt.xticks(np.arange(numBins))
    # plt.show()


################################################
#################### Tensorflow NN #############
################################################

if TRAIN_NN_TF:
    data, labels = load_data("../data/dialog_acts.dat")
    vectorizer = CountVectorizer()

    LABELS = np.unique(labels)
    labels = preprocess_labels(labels, LABELS)



    # The ratio of training data.
    split = .85

    split = int(split * data.shape[0])
    X_train = data[:split]
    y_train = labels[:split]
    X_test = data[split:]
    y_test = labels[split:]

    # Split the full dataset in a training part of 85% and a test part of 15%
    X_train, X_test, y_train, y_test = np.array(train_test_split(data, labels, test_size=0.15))

    X_train = vectorizer.fit_transform(X_train)
    X_train = X_train.toarray()

    X_test = vectorizer.transform(X_test).toarray()


    """
    Classify dialog acts via a simple Multilayer Perceptron (MLP) model
    """

    vocab_size = X_train.shape[1]

    # define network
    model = Sequential()
    model.add(Dense(512, input_shape=(vocab_size,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(LABELS), activation='sigmoid'))
    # compile network
    opt = Adam(learning_rate=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # fit network
    history = model.fit(X_train, y_train, epochs=10, verbose=1, validation_split=0.2)

    plot_training_history(history)

    # evaluate
    score = model.evaluate(X_test, y_test, verbose=1)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    print(f'Learning rate: {K.eval(model.optimizer.lr)}')

    # save the model
    model_json = model.to_json()
    with open("../data/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../data/model.h5")
    print("Saved model to disk")

    # save the vectorizer as pickle
    pickle.dump(vectorizer, open("../data/bag_of_words.pkl", "wb"))




################################################
#################### PyTorch NN ################
################################################

data, labels = load_data("../data/dialog_acts.dat")
SHOW_STATS = False
LABELS = np.unique(labels)
print(f"The labels in this dataset are: {LABELS}")

# The ratio of training data.
split = .85

split = int(split * data.shape[0])
X_train = data[:split]
y_train = labels[:split]
X_test = data[split:]
y_test = labels[split:]

class NeuralNetworkClassifier(Module):
    def __init__(self, vectorizer: CountVectorizer, tfidf, hidden_size: int):
        """
        Init.
        :param vectorizer: The vectorizer to use when converting the string input
        into bag of words representation.
        :param max_depth: The maximum depth of the tree.
        """
        super(NeuralNetworkClassifier, self).__init__()
        dim = vectorizer.transform(['x']).toarray().shape[1]
        self.net = Sequential(
            Linear(in_features=dim, out_features=512),
            ReLU(),
            Dropout(0.2),
            Linear(in_features=512, out_features=256),
            ReLU(),
            Dropout(0.2),
            Linear(in_features=256, out_features=len(LABELS))
        )
        self.vectorizer = vectorizer
        self.tfidf = tfidf

    def forward(self, x: np.ndarray):
        # vec = self.tfidf.transform(self.vectorizer.transform(x))
        vec = self.vectorizer.transform(x)
        vec = torch.tensor(vec.toarray(), dtype=torch.float32)
        return self.net(vec)

def train_nn(network: Module, optimizer, data: np.ndarray, golden_labels: np.ndarray) -> None:
    """
    Trains the network.
    :param network: pytorch Module to train.
    :param optimizer: The optimizer to use for training.
    :param data: Array of string utterances.
    :param golden_labels: Array of string labels.
    """
    network.train(True)
    target = torch.tensor(preprocess_labels(golden_labels, mapping=LABELS), dtype=torch.long)
    out = network(data)
    # Class weight is the inverse of frequency.
    sample_weights = np.unique(golden_labels, return_counts=True)[1]
    sample_weights = 1 / sample_weights
    loss = cross_entropy(input=out, target=target, weight=torch.tensor(sample_weights, dtype=torch.float32))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def test_nn(network: Module, data: np.ndarray, labels: np.ndarray)-> Tuple[float, np.ndarray]:
    """
    Evaluates the network on the dataset.
    :param network: pytorch Module.
    :param data: Array of string utterances.
    :param labels: Array of string labels.
    :return: A tuple of (accuracy, confusion matrix).
    """
    network.train(False)
    with torch.no_grad():
        pred = torch.argmax(network(data), dim=1, keepdim=False).numpy()
    labels = preprocess_labels(labels, mapping=LABELS)
    acc = accuracy_score(y_true=labels, y_pred=pred)
    conf_mat = confusion_matrix(y_true=labels, y_pred=pred)

    return acc, conf_mat


################################################
############## Evaluation ######################
################################################


majority_c = MajorityClassifier(training_labels=y_train)
acc, _ = test_classifier(classifier=majority_c, data=X_test, labels=y_test)
print(f"Accuracy of majority classifier: {acc*100:.3f}%")

keyword_c = KeywordClassifier("../data/kw.json")
acc, _ = test_classifier(classifier=keyword_c, data=X_test, labels=y_test)
print(f"Accuracy of keyword classifier: {acc * 100:.3f}%")

if TRAIN_NN_PYTORCH:
    neural_network_c = NeuralNetworkClassifier(vectorizer_train, tfidf, hidden_size=1024)
    optimizer = torch.optim.Adam(neural_network_c.parameters(), lr=0.001)
    epochs = 100

    print("Training neural network.")
    for e in range(1, epochs+1):
        train_nn(network=neural_network_c, data=X_train, golden_labels=y_train, optimizer=optimizer)
        acc, _ = test_nn(neural_network_c, X_test, y_test)
        print(f"== Epoch {e}/{epochs} completed. Accuracy on test set: {acc}")

    acc, _ = test_nn(neural_network_c, X_test, y_test)
    print(f"== Epoch {e}/{epochs} completed. Final accuracy on test set: {acc}")

    # Set your classifier here.
    CLF = neural_network_c


    def handle_input(s: str) -> None:
        s = s.lower()
        # Special case needed for the neural network.
        if isinstance(CLF, Module):
            with torch.no_grad():
                out = CLF([s])
            out = torch.argmax(out, dim=1)
        else:
            out = CLF([s])
        return LABELS[out.item()]


    while True:
        s = input("Prompt: ")
        if s == "exit":
            break
        print(f"Predicted label: {handle_input(s)}")

    torch.save(CLF, "ckpt.pyt")
    import pickle

    with open("preprocessing.utils", 'wb') as file:
        pickle.dump((vectorizer_train, tfidf), file)
