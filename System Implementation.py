import random
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.keras.models import model_from_json
import pandas
######
from typing import List, Tuple, Union
import numpy as np
import json
from abc import ABC, abstractmethod
import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import sklearn.tree as sk_tree
from sklearn.model_selection import train_test_split

import torch
from torch.nn import Module, Linear, ReLU, Sequential
from torch.nn.functional import cross_entropy

import keras.backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential, save_model, load_model

import nltk
from nltk.corpus import stopwords

import pickle

from collections import Counter
from collections import defaultdict

# for downloading dataset
from urllib.request import urlretrieve
import re
#!pip install python-levenshtein
import pandas
from Levenshtein import distance as levenshtein_distance

#####################################################
# IMORTANT VARIABLES TO CONTROL HOW THE PROGRAM RUNS
first_run = True
interactive_part = True
#####################################################
def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the data file from the provided path.
    Returns a tuple of numpy arrays, each with shape (N,) where N is
    the batch size. The first element contains a list of utterances, the
    second element contains the corresponding labels (as string).
    """

    DSTC2_url = "https://drive.google.com/uc?export=download&id=1bQ9EX8-W_Bo991qrt96j6xQtxYeYk5aR"
    path = urlretrieve(DSTC2_url, "DSTC2.dat")[0]

    with open(path, 'r') as file:
        lines = file.readlines()
    # Convert the data to lower case for training and testing, this will increase the accuracy of the classifier
    data = list(map(lambda s: [s.split()[0].lower(), ' '.join(s.split()[1:]).lower()], lines))
    data = np.array(data)

    return data[:, 1], data[:, 0]

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

TEST_SIZE = 0.15
STOP_WORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

features, labels = load_data("DSTC2.dat")

# Split the full dataset in a training part of 85% and a test part of 15%
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=TEST_SIZE)

# Now we obtain a list of all labels that are used in the dataset. Which are:
# ['ack' 'affirm' 'bye' 'confirm' 'deny' 'hello' 'inform' 'negate' 'null'
#  'repeat' 'reqalts' 'reqmore' 'request' 'restart' 'thankyou']
LABELS = np.unique(labels)

class MajorityClassifier:
    """
    This classifier always returns the label for the majority class.
    """
    def __init__(self):
        """
        Init.
        :param return_value: The value to return. Should be the
        index of the majority class in the dataset.
        """
        c = Counter(y_train);
        most_common_label = c.most_common(1)[0][0]
        self.return_value = LABELS.tolist().index(most_common_label)

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

        json_url = "https://drive.google.com/uc?export=download&id=1tYpVxRjvD8ahNGe_YJONFIq7G7U65eiC"
        path = urlretrieve(json_url, "kw.json")[0]

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

# Train rule-based baseline model
acts_keywords = {}

def train_rule_based():
    for i, words in enumerate(X_train):
        act = list(y_train)[i]
        if act not in acts_keywords:
            acts_keywords[act] = defaultdict()

        for keyword in words:
            if keyword in STOP_WORDS:
                continue
            else:
                if keyword not in acts_keywords[act]:
                    acts_keywords[act][keyword] = 1
                else:
                    acts_keywords[act][keyword] += 1
if first_run:
    train_rule_based()

def rule_based(utterance):
    act_count = defaultdict(lambda: 0)

    for word in utterance:
        if word in STOP_WORDS:
            continue
        else:
            for act_name, act in acts_keywords.items():
                if word in act:
                    act_count[act_name] += act[word]
                    continue

    acts = sorted(act_count.items(), key=lambda item: item[1], reverse=True)
    if not acts:
        prediction = "inform"
    else:
        prediction = acts[0][0]

    return prediction

def evaluate_rule_based():
    true_positives = 0

    for i, utterance in enumerate(list(X_test)):
        predicted_label = rule_based(utterance)
        golden_label = list(y_test)[i]

        if predicted_label == golden_label:
            true_positives += 1

    accuracy = true_positives / len(list(y_test))
    return round(accuracy, 2)

if first_run:
    rule_based_accuracy = evaluate_rule_based()
#print(f'Rule-based system [accuracy] = {rule_based_accuracy}')

class DecisionTreeClassifier:
    def __init__(self, vectorizer: CountVectorizer, max_depth: Union[int, None] = None):
        """
        Init.
        :param vectorizer: The vectorizer to use when converting the string input
        into bag of words representation.
        :param max_depth: The maximum depth of the tree.
        """
        self.vectorizer = vectorizer
        self.clf = sk_tree.DecisionTreeClassifier(max_depth=max_depth)

    def __call__(self, texts: np.ndarray):
        vec = self.vectorizer.transform(texts)
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
    def __init__(self, vectorizer: CountVectorizer, hidden_size: int):
        """
        Init.
        :param vectorizer: The vectorizer to use when converting the string input
        into bag of words representation.
        :param max_depth: The maximum depth of the tree.
        """
        super(NeuralNetworkClassifier, self).__init__()
        dim = vectorizer.transform(['x']).toarray().shape[1]
        self.net = Sequential(
            Linear(in_features=dim, out_features=hidden_size),
            ReLU(),
            Linear(in_features=hidden_size, out_features=len(LABELS))
        )
        self.vectorizer = vectorizer

    def forward(self, x: np.ndarray):
        vec = self.vectorizer.transform(x)
        vec = torch.tensor(vec.toarray(), dtype=torch.float32)
        return self.net(vec)

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

    predictions = classifier(data)
    labels = preprocess_labels(labels, mapping=LABELS)
    acc = accuracy_score(y_true=labels, y_pred=predictions)
    conf_mat = confusion_matrix(y_true=labels, y_pred=predictions)

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

#The Majority classifier
majority_c = MajorityClassifier()
acc, _ = test_classifier(classifier=majority_c, data=X_test, labels=y_test)
#print(f"Accuracy of majority classifier: {acc*100:.3f}%")

#The Keyword classifier
keyword_c = KeywordClassifier("kw.json")
acc, _ = test_classifier(classifier=keyword_c, data=X_test, labels=y_test)
#print(f"Accuracy of keyword classifier: {acc * 100:.3f}%")


#For the advanced models we need to setup a vectorizer first,
# that converts the strings to bag of words vectors. It is fit on the whole dataset.
vectorizer = CountVectorizer()
_ = vectorizer.fit_transform(features)

#Decision tree
decision_tree_c = DecisionTreeClassifier(vectorizer, max_depth=15)

#print("Training decision tree classifier.")
sample_weights = np.unique(y_train, return_counts=True)[1]
sample_weights = 1 / sample_weights
sample_weights = list(zip(range(0, len(sample_weights)), sample_weights))
sample_weights = compute_sample_weight(dict(sample_weights), preprocess_labels(y_train, mapping=LABELS))

decision_tree_c.fit(X_train, y_train, sample_weights)
acc, conf_mat = test_classifier(decision_tree_c, X_test, y_test)
#print(f"Decision tree trained. Accuracy on training set: {100 * acc:.3f}%")


#TRAIN NN


#Train Neural Network
# Constants
STOP_WORDS = stopwords.words("english")
DIALOG_ACTS = ["ack","affirm","bye","confirm","deny","hello", "inform","negate","null",
               "repeat","reqalts","reqmore","request","restart","thankyou"]
CUT_OFF = 10
FILEPATH = './saved_model'


# Variables
features = []
labels = []

with open("DSTC2.dat", 'r') as f:
    for line in f:
        tokens = line.strip().lower().split(" ")

        words = tokens[1:]
        label = tokens[0]

        words = [word for word in words if word not in STOP_WORDS]

        features.append(words)
        labels.append(label)


# tranform list of words to string of words
features_BOW = [" ".join(x) for x in features][CUT_OFF:]

labels = [DIALOG_ACTS.index(label) for label in labels][CUT_OFF:]
golden_labels = np.asarray(labels)

vectorizer = CountVectorizer(
    analyzer="word", tokenizer=nltk.word_tokenize,
    preprocessor=None, stop_words='english', max_features=None)

bag_of_words = vectorizer.fit_transform(features_BOW)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(features_BOW))
pickle.dump(vectorizer.vocabulary_, open("feature.pkl", "wb"))

bag_of_words = np.asarray(tfidf.toarray())

# Split the full dataset in a training part of 85% and a test part of 15%
X_train, X_test, y_train, y_test = np.array(train_test_split(bag_of_words, golden_labels, test_size=0.15))

"""
Classify dialog acts via a simple Multilayer Perceptron (MLP) model
"""

vocab_size = bag_of_words.shape[1]
if first_run:
    # define network
    model = Sequential()
    model.add(Dense(512, input_shape=(vocab_size,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(DIALOG_ACTS), activation='sigmoid'))
    # compile network
    opt = Adam(learning_rate=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # fit network
    model.fit(X_train, y_train, epochs=30, verbose=1, validation_split=0.2)

    # evaluate
    score = model.evaluate(X_test, y_test, verbose=1)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    print(f'Learning rate: {K.eval(model.optimizer.lr)}')

    # Save the model
    save_model(model, FILEPATH)

# Load the model
model = load_model(FILEPATH, compile=True)

transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("feature.pkl", "rb")))

#Interactive part
# Set your classifier here.
CLF = model

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


while interactive_part:
    s = input("Type 'STOP' if you do not want to enter new instance, else enter new sentence:")
    if s == "STOP":
        interactive_part = False

    else:
        s = input("Please enter a new sentence: ")
        print(f"Predicted label: {handle_input(s)}")



# load json and create model
#json_file = open('data/model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
# load weights into new model
#model.load_weights("data/model.h5")

############################### PART 1B DIALOG MANAGEMENT

restaurant_info = pandas.read_csv("data/restaurant_info.csv")

# Constants
MIN_LEVENSHTEIN_DISTANCE = 3

BASELINE_DIALOG_ACT = "inform"

# the knowledge structure representing the dialog state/frame
dialog_state = {
    "slots": [
        {
            "name": "area",
            "filler": "dontcare",
            "question": "What part of town do you have in mind?",
            "confirmed": True
        },
        {
            "name": "food",
            "filler": "dontcare",
            "question": "What kind of food would you like?",
            "confirmed": True
        },
        {
            "name": "pricerange",
            "filler": "dontcare",
            "question": "What kind of price range are you looking for?",
            "confirmed": True
        }
    ],
    "matched_restaurants": [],
    "alternative_restaurants": [],
    "system_utterances": [],
    "user_utterance": "",
    "states": [],
    "end_conversation" : False
}

domain_terms = {
    "food": list(restaurant_info["food"].dropna().unique()),  # a list of food types that are represented in the data
    "area": list(restaurant_info["area"].dropna().unique()),  # a list of area's that are represented in the data
    "pricerange": list(restaurant_info["pricerange"].dropna().unique())     # a list of price ranges that are represented in the data
}


# Utility functions

def extract_dialog_act(user_input):
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array([user_input])))
    bag_of_words = np.asarray(tfidf.toarray())

    # Generate predictions for samples
    predictions = model.predict(bag_of_words)
    return predictions


def extract_preferences(dialog_state):
    """
    Looks for food type, area and price range in the given sentence with a keyword-matching algorithm
    And fills slots of the frame if preferences are found
    :param dialog_state:
    :return:
    """

    preferences = defaultdict(lambda: "dontcare")
    current_state = "" if not dialog_state["states"] else dialog_state["states"][0]

    regexes = {
        "area": [
            domain_terms["area"],
            "(?<=restaurant in the\s)(\w+)",
            "(any) area",
            "(any) part"
        ],
        "food": [
            domain_terms["food"],
            "(\w+)(?=\s+food)",

        ],
        "pricerange": [
            domain_terms["pricerange"],
            "(?<=a\s)(\w+)(?=\srestaurant)"
        ]
    }
    # for each regular expression defined per domain search for a match with the current user utterance
    for slot_filler, slot_regexes in regexes.items():
        for slot_regex in slot_regexes:
            if type(slot_regex) is list:
                for regex in slot_regex:
                    match = re.search("(" + regex + ")", dialog_state["user_utterance"])
                    if match:
                        break
            else:
                match = re.search(slot_regex, dialog_state["user_utterance"])

            if match: # if a match is found, save the match in the corresponding slot in preferences
                preferences[slot_filler] = match.group(1)
                break

    # Miscellaneous cases (e.g. any)
    if re.search("^any$|^it doesnt matter$", dialog_state["user_utterance"]):
        if ("food" in current_state):
            preferences["food"] = "dontcare"
        elif ("area" in current_state):
            preferences["area"] = "dontcare"
        elif ("price" in current_state):
            preferences["pricerange"] = "dontcare"

    # save cases with 'any' in the utterances as dontcare
    for pref_name, pref_value in preferences.items():
        if pref_value == "any":
            preferences[pref_name] = "dontcare"

    # If user utterance is only one word, check for minimal edit distance with domain terms (e.g. "spenish")
    if len(dialog_state["user_utterance"].split()) == 1:
        preferences = levenshtein_edit_distance(dialog_state["user_utterance"], ["food", "area", "pricerange"])

    return preferences


def levenshtein_edit_distance(pref_value, domains):
    """
    Use Levenshtein edit distance as implemented in the python-Levenshtein library to map values
    to the closest domain term in case an exact match is not found.
    :return:
    """
    best_match = []

    #for domain in domain_terms[pref_name]:
    for domain in domains:
        for term in domain_terms[domain]:
            distance = levenshtein_distance(pref_value, term)   # calculate the levenshtein_distance from the pref_value to every term in the database
            if distance <= MIN_LEVENSHTEIN_DISTANCE:
                min_distance = distance
                best_match.append({"term": term,                # save a term with a distance that is smaller than MIN_LEVENSHTEIN_DISTANCE to best_match
                                   "distance": distance,        # with its domain
                                   "domain": domain})

    if not best_match:
        return False
    else:
        best_match = list(filter(lambda d: d["distance"] == min_distance, best_match)) # extract the terms with the lowest distance
        random.shuffle(best_match)  #randomly choose one of the best matches

    return {best_match[0]["domain"]: best_match[0]["term"]}


def rule_based_dialog_classifier(user_utterance):
    """
    Looks for certain words or phrases to deduce which dialog
    act should be given to the user utterance
    """

    dialog_act = ""

    regexes = {
        "inform": [
            "any part of town",
            "^any$",
            "^i want a restaurant",
            "^im looking for"
        ],
        "restart": [
            "reset",
            "start over"
        ],
        "bye": [
            "bye",
            "goodbye"
        ],
        "deny":[
            "no",
            "wrong"
        ],
        "confirm":[
            "yes",
            "that is right"
        ]
    }

    for act, regexes in regexes.items():
        for regex in regexes:
            if re.search(regex, user_utterance):
                return act

    if dialog_act == "":
        return BASELINE_DIALOG_ACT


# https://queirozf.com/entries/pandas-query-examples-sql-like-syntax-queries-in-dataframes
def query_restaurant_info(query):
    """
    Query restaurant_info database
    :return: list of dicts representing found records in database
    """
    results = restaurant_info.query(query).to_dict('records')
    return results


def check_restaurant_availability(dialog_state):
    # Create a list of filters
    query = ""
    restaurant_filters = []
    for slot in dialog_state["slots"]:
        if slot["filler"] != "dontcare":
            restaurant_filters.append(f'{slot["name"]} == \'{slot["filler"]}\'')

    # Build query to find matching restaurants
    if not restaurant_filters:
        # all restaurants
        query = "ilevel_0 in ilevel_0"
    else:
        query = "(" + ") and (".join(restaurant_filters) + ")"

    dialog_state["matched_restaurants"] = query_restaurant_info(query=query)

    # Decide based on number of found matching restaurants
    num_matched_restaurants = len(dialog_state["matched_restaurants"])

    # Assignment: If the number of restaurants satisfying the current set of preferences is 0, then the user should be informed and given the option to retract and restate any number of previously stated preferences.
    if num_matched_restaurants == 0:
        # Assigment: If the number of restaurants satisfying the current set of preferences is 0 then the system should offer alternatives for the conflicting preferences.
        # Assigmment: TODO The alternatives should be modeled using a set membership function. Of course an alternative should only be offered if the new set of preferences is satisfiable in the database.

        random.shuffle(restaurant_filters)
        del restaurant_filters[0]
        query = "(" + ") and (".join(restaurant_filters) + ")"
        dialog_state["alternative_restaurants"] = query_restaurant_info(query)
        restaurant_one = dialog_state["alternative_restaurants"][0]
        restaurant_two = dialog_state["alternative_restaurants"][1]

        system_utterance = f"""there are no suggestions that satisfy your preferences.
              The following restaurants are available:
          1. Restaurant {restaurant_one["restaurantname"]} serving {restaurant_one["food"]} in {restaurant_one["area"]} part of town for {restaurant_one["price_type"]}
          2. Restaurant {restaurant_two["restaurantname"]} serving {restaurant_two["food"]} in {restaurant_two["area"]} part of town for {restaurant_two["price_type"]}
          Do you want to:
          a. change your preferences
          b. choose one of these alternatives?
          """

        dialog_state["system_utterances"].insert(system_utterance, 0)  # TODO: Improve
        dialog_state["states"].insert(0, f"Suggest restaurant")


    # Assignment: If the number of restaurants satisfying the current set of preferences is 1, then the system should not ask any remaining preferences and should immediately present the recommendation.
    elif num_matched_restaurants == 1:
        dialog_state["system_utterances"].insert(0, "You can eat Bart's ... ")  # TODO: Improve

    # Assigment: If the number of restaurants satisfying the current set of preferences is 2 or more then the system should proceed normally.
    elif num_matched_restaurants > 1:
        for slot in dialog_state["slots"]:
            if slot["filler"] == "dontcare":
                dialog_state["system_utterances"].insert(0, slot["question"])
                dialog_state["states"].insert(0, f"ask {slot['name']}")
                break

    return dialog_state

def inform_response(dialog_state):
    """
    Create response in case of inform dialog act
    :param dialog_state:
    :return: dialog_state
    """
    needs_confirm = False
    system_utterance = ""
    preferences = extract_preferences(dialog_state)

    # Apply Levenshtein Edit Distance if no exact match with domain terms is found
    for pref_name, pref_value in preferences.items():
        if pref_value not in domain_terms[pref_name] and pref_value != "dontcare":
            levenshtein = levenshtein_edit_distance(pref_value=pref_value, domains =[pref_name])

            if levenshtein == False: #if no match is found in the database, return feedback to user
                dialog_state["system_utterances"].insert(0,
                                                         f'I am sorry but there is no restaurant with {pref_value} {pref_name}.')
                return dialog_state
            else:
                needs_confirm = True

                preferences[pref_name] = levenshtein[pref_name]
                #return dialog_state
               # preferences[levenshtein["domain"]] = levenshtein["term"]

    # Fill slots
    for slot in dialog_state["slots"]:
        for pref_name, pref_value in preferences.items():
            if slot["name"] == pref_name:
                slot["filler"] = pref_value
                break



    # check if there are still restaurants avaiable with the current preferences
    dialog_state = check_restaurant_availability(dialog_state)

    if needs_confirm:
        dialog_state["system_utterances"].insert(0,
                                                 f'Did you mean a restaurant with {levenshtein[pref_name]} {pref_name}?')
        for slot in dialog_state["slots"]:
            if slot["name"] == pref_name:
                slot["confirmed"] = False
        needs_confirm = False
    return dialog_state

def bye_response(dialog_state):
    dialog_state["end_conversation"] = True

    return dialog_state

def confirm_response(dialog_state):
    for slot in dialog_state["slots"]:
        if not slot["confirmed"]:
            slot["confirmed"] = True

    for slot in dialog_state["slots"]:
        if slot["filler"] == "dontcare":
            dialog_state["system_utterances"].insert(0, slot["question"])
            dialog_state["states"].insert(0, f"ask {slot['name']}")

    return dialog_state

    return dialog_state

def deny_response(dialog_state):
    for slot in dialog_state["slots"]:
        if not slot["confirmed"]:
            slot["filler"] = "dontcare"
            slot["confirmed"] = True
            dialog_state["system_utterances"].insert(0, slot["question"])
            dialog_state["states"].insert(0, f"ask {slot['name']}")
    #print(dialog_state)
    return dialog_state

def state_transition(dialog_state):
    """
    The main dialog manager though which everything runs
    :param dialog_state:
    :return:
    """

    system_utterance = ""

    dialog_act_softmax = extract_dialog_act(dialog_state["user_utterance"])
    dialog_act = DIALOG_ACTS[np.argmax(dialog_act_softmax)]
    dialog_act_prob = dialog_act_softmax.max() # gives the probability of the dialog acts with the highest probability

    if dialog_act_prob < 0.8: # if the probability is low, check with a rule based classifier if we can classify it correctly
        dialog_act = rule_based_dialog_classifier(dialog_state["user_utterance"])
        dialog_act_prob = 1

    # respond to the different kinds of dialog acts
    if dialog_act == "inform":
        dialog_state = inform_response(dialog_state)

    elif dialog_act == "ack":
        pass  # ack_response(dialog_state) TODO

    elif dialog_act == "affirm":
        dialog_state = confirm_response(dialog_state)

    elif dialog_act == "bye":
        dialog_state = bye_response(dialog_state)

    elif dialog_act == "confirm":
        dialog_state = confirm_response(dialog_state)

    elif dialog_act == "deny":
        dialog_state= deny_response(dialog_state)

    elif dialog_act == "hello":
        pass  # hello_response(dialog_state) TODO

    elif dialog_act == "negate":
        pass  # negate_response(dialog_state) TODO

    elif dialog_act == "null":
        pass

    elif dialog_act == "repeat":
        pass  # repeat_response(dialog_state) TODO

    elif dialog_act == "reqalts":
        pass  # reqalt_response(dialog_state) TODO

    elif dialog_act == "reqmore":
        pass  # reqmore_response(dialog_state) TODO

    elif dialog_act == "request":
        pass  # request_response(dialog_state) TODO

    elif dialog_act == "restart":
        pass

    elif dialog_act == "thankyou":
        pass  # thankyou_response(dialog_state) TODO

    return dialog_state


# the system-user interaction in the python console
counter = 0
conversation = True
try:
    while True:
        #print(dialog_state["slots"])
        welcome_message = "System: Hello, welcome to the Cambridge restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?"
        if counter == 0:
            dialog_state["user_utterance"] = input(welcome_message).lower()
            dialog_state = state_transition(dialog_state)
            counter += 1
            continue

        # if there are no system utterances available, use welcome message
        if dialog_state["end_conversation"]:
            sys.exit("Thanks and goodbye!")
            #break
        if not dialog_state["system_utterances"]:
            dialog_state["user_utterance"] = input(welcome_message)
        else:
            dialog_state["user_utterance"] = input(f'System: {dialog_state["system_utterances"][0]}')


        dialog_state = state_transition(dialog_state)
        print("jeeh")
        counter += 1

except EOFError as e:
    sys.exit("Thanks and goodbye!")
