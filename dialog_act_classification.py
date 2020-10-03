from typing import List, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import sklearn.tree as sk_tree
from abc import ABC, abstractmethod
import json
import torch
from torch.nn import Module, Linear, ReLU, Sequential, Sigmoid, Dropout
from torch.nn.functional import cross_entropy


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
    return out


# Loading the data.
data, labels = load_data("data/dialog_acts.dat")

# The ratio of training data.
split = .85

split = int(split * data.shape[0])
training_data = data[:split]
training_labels = labels[:split]
test_data = data[split:]
test_labels = labels[split:]


LABELS = np.unique(labels)
print(f"The labels in this dataset are: {LABELS}")


# Defining the classifiers
class MajorityClassifier:
    """
    This classifier always returns the label for the majority class.
    """
    def __init__(self, return_value: int):
        """
        Init.
        :param return_value: The value to return. Should be the
        index of the majority class in the dataset.
        """
        self.return_value = return_value

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
                if any([w in s.split() for w in self.keywords[k]]):
                    return k
            # If no keyword is matched return null.
            return 'null'

        predictions = [find_keyword(s) for s in texts]
        return np.array([self.label_names.index(p) for p in predictions])


class DecisionTreeClassifier:
    def __init__(self, vectorizer: CountVectorizer, tfidf, max_depth: Union[int, None] = None):
        """
        Init.
        :param vectorizer: The vectorizer to use when converting the string input
        into bag of words representation.
        :param max_depth: The maximum depth of the tree.
        """
        self.vectorizer = vectorizer
        self.tfidf = tfidf
        self.clf = sk_tree.DecisionTreeClassifier(max_depth=max_depth)

    def __call__(self, texts: np.ndarray):
        vec = self.tfidf.transform(self.vectorizer.transform(texts))
        vec = np.array(vec.toarray())
        out = self.clf.predict(vec)
        return out

    def fit(self, x, y, sample_weights):
        """
        Fits the tree.
        :param x: The data to fit on.
        :param y: The labels of the data.
        :param sample_weights: The sample weights to use in case
        of imbalanced classes.
        """
        x = preprocess_data(x, self.vectorizer)
        y = preprocess_labels(y, mapping=LABELS)
        self.clf.fit(x, y, sample_weight=sample_weights)


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
        vec = self.tfidf.transform(self.vectorizer.transform(x))
        vec = torch.tensor(vec.toarray(), dtype=torch.float32)
        return self.net(vec)


# Helper functions for training and testing.
def test_classifier(classifier, data: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Evaluates the classifier on the given batch. Outputs the accuracy and the confusion matrix.
    :param classifier: The classifier to evaluate.
    :param data: The test data.
    :param labels: Test labels.
    :return: A tuple of (accuracy, confusion matrix).
    """
    # If the classifier is a torch module the input type should be a tensor
    # and not a numpy array.
    if isinstance(classifier, Module):
        data = torch.tensor(data, dtype=torch.float32)

    pred = classifier(data)
    labels = preprocess_labels(labels, mapping=LABELS)
    acc = accuracy_score(y_true=labels, y_pred=pred)
    conf_mat = confusion_matrix(y_true=labels, y_pred=pred)

    return acc, conf_mat


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


def train_nn(network: Module, optimizer, data: np.ndarray, labels: np.ndarray) -> None:
    """
    Trains the network.
    :param netowrk: pytorch Module to train.
    :param optimizer: The optimizer to use for training.
    :param data: Array of string utterances.
    :param labels: Array of string labels.
    """
    network.train(True)
    target = torch.tensor(preprocess_labels(labels, mapping=LABELS), dtype=torch.long)
    out = network(data)
    # Class weight is the inverse of frequency.
    sample_weights = np.unique(labels, return_counts=True)[1]
    sample_weights = 1 / sample_weights
    loss = cross_entropy(input=out, target=target, weight=torch.tensor(sample_weights, dtype=torch.float32))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Testing the models
majority_c = MajorityClassifier(return_value=LABELS.tolist().index('inform'))
acc, _ = test_classifier(classifier=majority_c, data=test_data, labels=test_labels)
print(f"Accuracy of majority classifier: {acc*100:.3f}%")


keyword_c = KeywordClassifier("data/kw.json")
acc, _ = test_classifier(classifier=keyword_c, data=test_data, labels=test_labels)
print(f"Accuracy of keyword classifier: {acc * 100:.3f}%")


# Preprocessing tools for advanced methods:
vectorizer = CountVectorizer()
tfidf = TfidfTransformer()
_ = tfidf.fit_transform(vectorizer.fit_transform(data))


decision_tree_c = DecisionTreeClassifier(vectorizer, tfidf, max_depth=30)

print("Training decision tree classifier.")

sample_weights = np.unique(training_labels, return_counts=True)[1]
sample_weights = 1 / sample_weights
sample_weights = list(zip(range(0, len(sample_weights)), sample_weights))
sample_weights = compute_sample_weight(dict(sample_weights), preprocess_labels(training_labels, mapping=LABELS))

decision_tree_c.fit(training_data, training_labels, sample_weights)
acc, conf_mat = test_classifier(decision_tree_c, test_data, test_labels)
print(f"Decision tree trained. Accuracy on training set: {100 * acc:.3f}%")


neural_network_c = NeuralNetworkClassifier(vectorizer, tfidf, hidden_size=1024)
optimizer = torch.optim.Adam(neural_network_c.parameters(), lr=0.001)
epochs = 100

print("Training neural network.")
for e in range(1, epochs+1):
    train_nn(network=neural_network_c, data=training_data, labels=training_labels, optimizer=optimizer)
    acc, _ = test_nn(neural_network_c, test_data, test_labels)
    print(f"== Epoch {e}/{epochs} completed. Accuracy on test set: {acc}")

# ####################### Interactive mode #################################

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


print("This is the interactive part. You can test the neural network classifier by typing an utterance. Type 'exit' to exit the program.")
while True:
    s = input("User input: ")
    if s == "exit":
        break
    print(f"Predicted label: {handle_input(s)}")
