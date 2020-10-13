import pickle
import sys
import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import torch
from torch.nn import Module, ReLU, Linear, Sequential, Dropout

import copy
import pandas
# terminal: pip install python-levenshtein
from Levenshtein import distance as levenshtein_distance
from collections import defaultdict
import time
import json


class Antecedent:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"{self.name, self.value}"

    def __str__(self):
        return self.__repr__()


class Rule:
    def __init__(self, identifier, antecedents, consequent, truth_value):
        self.identifier = identifier
        self.antecedents = antecedents
        self.consequent = consequent
        self.truth_value = truth_value

    def __repr__(self):
        return f"{self.identifier}. {self.antecedents} => ({self.consequent}, {self.truth_value})"

    def __str__(self):
        return self.__repr__()


RULES_ = [
    Rule(1, [Antecedent("pricerange", "cheap"), Antecedent("food", "good")], "busy", True),
    Rule(2, [Antecedent("food", "spanish")], "long_time", True),
    Rule(3, [Antecedent("busy", True)], "long_time", True),
    Rule(4, [Antecedent("long_time", True)], "children", False),
    Rule(5, [Antecedent("busy", True)], "romantic", False),
    Rule(6, [Antecedent("long_time", True)], "romantic", True),
    # New rules:
    Rule(7, [Antecedent("busy", True)], "popular", True),
    Rule(8, [Antecedent("children", True)], "romantic", False),
    Rule(9, [Antecedent("busy", True), Antecedent("children", True)], "loud", True),
    Rule(10, [Antecedent("loud", True)], "romantic", False),
    Rule(11, [Antecedent("busy", False)], "loud", False),
    Rule(12, [Antecedent("food", "french")], "romantic", True)
]


def is_applicable(rule, restaurant):
    """
    Predicate function, returns whether a rule is applicable
    to the given restaurant.
    """
    fields = restaurant.keys()
    for a in rule.antecedents:
        if a.name not in fields or a.value != restaurant[a.name] or rule.consequent in fields:
            return False
    return True


def apply_inference(restaurant, rules):
    """
    Applies the provided rules to the restaurant instance.
    :param restaurant: A dictionary representing a restaurant.
    :param rules: The set of rules to be applied.
    :return: The restaurant dictionary with additional fields that
    are the result of the implications.
    """
    applicable = [...]
    while applicable:
        applicable = [r for r in rules if is_applicable(r, restaurant)]
        for rule in applicable:
            restaurant[rule.consequent] = rule.truth_value
    return restaurant


# The database
restaurant_info = pandas.read_csv("data/restaurant_info.csv")

# Constants
MIN_LEVENSHTEIN_DISTANCE = 3

BASELINE_DIALOG_ACT = "inform"
DIALOG_ACTS = [
    "ack",
    "affirm",
    "bye",
    "confirm",
    "deny",
    "hello",
    "inform",
    "negate",
    "null",
    "repeat",
    "reqalts",
    "reqmore",
    "request",
    "restart",
    "thankyou"
]

LABELS = ['ack' 'affirm' 'bye' 'confirm' 'deny' 'hello' 'inform' 'negate' 'null'
          'repeat' 'reqalts' 'reqmore' 'request' 'restart' 'thankyou']

# the knowledge structure representing the dialog state/frame
original_state = {
    "values": {
        "food": None,
        "area": None,
        "pricerange": None
    },
    "confident": {
        "food": True,
        "area": True,
        "pricerange": True
    },
    # This contains a list of suitable restaurants with the preferences above.
    "suitable_restaurants": [],
    # This is the index of suggested restaurant in the list. This is needed because
    # the user might ask for alternatives.
    "current_index": 0,
    "done": False,
    "classifier": "nn",
    "delay": False,
    "allcaps": False
}

domain_terms = {
    "food": list(restaurant_info["food"].dropna().unique()) + ['dontcare'],
    # a list of food types that are represented in the data
    "area": list(restaurant_info["area"].dropna().unique()) + ['dontcare'],
    # a list of area's that are represented in the data
    "pricerange": list(restaurant_info["pricerange"].dropna().unique()) + ['dontcare']
    # a list of price ranges that are represented in the data
}

# Alternative preference sets.
ALTS = {
    "food": [
        ["thai", "chinese", "korean", "vietnamese", "asian" "oriental"],
        ["mediterranean", "spanish", "portuguese", "italian", "romanian", "tuscan", "catalan"],
        ["french", "european", "bistro", "swiss", "gastropub", "traditional"],
        ["north" "american", "steakhouse", "british"],
        ["lebanese", "turkish", "persian"],
        ["international", "modern" "european", "fusion"]
    ],
    "pricerange": [
        ["cheap", "moderate"],
        ["moderate", "expensive"]
    ],
    "area": [
        ["centre", "north", "west"],
        ["centre", "north", "east"],
        ["centre", "south", "west"],
        ["centre", "south", "east"]
    ]
}


# Loading the neural network classifier trained during 1a.
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
        self.label_names = LABELS

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


neural_network = torch.load("data/ckpt.pyt")
baseline = KeywordClassifier("data/kw.json")


# Utility functions

def extract_dialog_act(dialog_state, user_input):
    if dialog_state["classifier"] == "nn":
        with torch.no_grad():
            out = neural_network([user_input])
        return DIALOG_ACTS[torch.argmax(out, dim=1).item()]
    elif dialog_state["classifier"] == "baseline":
        return baseline(user_input)


def extract_preferences(user_utterance):
    """
    Looks for food type, area and price range in the given sentence with a keyword-matching algorithm
    And fills slots of the frame if preferences are found
    :param dialog_state:
    :return:
    """

    preferences = defaultdict(lambda: "dontcare")
    #     current_state = "" if not dialog_state["states"] else dialog_state["states"][0]

    regexes = {
        "area": [
            domain_terms["area"],
            "(?<=restaurant in the\s)(\w+)",
            "(any) area",
            "(any) part*",
            "(\w+)(?=\s+part)",
        ],
        "food": [
            domain_terms["food"],
            "(\w+)(?=\s+food)",

        ],
        "pricerange": [
            domain_terms["pricerange"],
            "(?<=a\s)(\w+)(?=\srestaurant)",
            "(any) range",
            "(any) price"
        ]
    }
    # for each regular expression defined per domain search for a match with the current user utterance
    for slot_filler, slot_regexes in regexes.items():
        for slot_regex in slot_regexes:
            if type(slot_regex) is list:
                for regex in slot_regex:
                    match = re.search("(" + regex + ")", user_utterance)
                    if match:
                        break
            else:
                match = re.search(slot_regex, user_utterance)

            if match:  # if a match is found, save the match in the corresponding slot in preferences
                preferences[slot_filler] = match.group(1)
                break

    # save cases with 'any' in the utterances as dontcare
    for pref_name, pref_value in preferences.items():
        if pref_value == "any":
            preferences[pref_name] = "dontcare"
    # Check with levenshtein whether the spelling is correct. If not, it will be flagged.
    confidence = {}
    preferences_ = {}
    for k, v in preferences.items():
        out, distance = levenshtein_edit_distance(v, k)
        if out:
            preferences_[k] = out[k]
            confidence[k] = distance == 0
    return dict(preferences_), confidence


def levenshtein_edit_distance(pref_value, domain):
    """
    Use Levenshtein edit distance as implemented in the python-Levenshtein library to map values
    to the closest domain term in case an exact match is not found.
    :return:
    """
    best_match = []
    min_distance = MIN_LEVENSHTEIN_DISTANCE
    for term in domain_terms[domain]:
        distance = levenshtein_distance(pref_value,
                                        term)  # calculate the levenshtein_distance from the pref_value to every term in the database
        if distance <= MIN_LEVENSHTEIN_DISTANCE:
            min_distance = min(distance, min_distance)
            best_match.append({"term": term,
                               # save a term with a distance that is smaller than MIN_LEVENSHTEIN_DISTANCE to best_match
                               "distance": distance,  # with its domain
                               "domain": domain})

    if not best_match:
        return False, False
    else:
        best_match = list(
            filter(lambda d: d["distance"] == min_distance, best_match))  # extract the terms with the lowest distance
        random.shuffle(best_match)  # randomly choose one of the best matches

    return {best_match[0]["domain"]: best_match[0]["term"]}, min_distance


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
        "deny": [
            "no",
            "wrong"
        ],
        "confirm": [
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


def generate_query(state):
    """
    Generates a database query based on the preferences contained in
    state.
    """
    q = []
    for k in ["food", "area", "pricerange"]:
        v = state["values"][k]
        c = state["confident"][k]
        if v is not None and v != "dontcare" and c:
            q.append(f"({k}=='{v}')")
    q = " and ".join(q)
    return q


def generate_question(dialog_state):
    """
    Generates the next slot filling question based on the state.
    Returns None if all slots filled with confident values.
    """
    msg = None
    # If something is already suggested no need to ask further questions.
    if len(dialog_state["suitable_restaurants"]) > 0:
        name = dialog_state["suitable_restaurants"][dialog_state["current_index"]]["restaurantname"]
        return f"You can ask more information about {name} if you are interested."

    # Checks for unsure values.
    unsure = [(k, dialog_state["values"][k]) for k, v in dialog_state["confident"].items() if not v]
    if len(unsure) > 0:
        msg = f"Please confirm that I understood you correctly: {unsure[0][0]} to be {unsure[0][1]}?"

    elif not dialog_state["values"]["food"]:
        msg = "What kind of food do you want to eat?"
    elif not dialog_state["values"]["area"]:
        msg = "What part of town do you have in mind?"
    elif not dialog_state["values"]["pricerange"]:
        msg = "What price range?"
    return msg


def get_alternatives_msg(dialog_state):
    """
    This function will offer an alternative for one of the preferences in case
    there is no restaurant for the current set of preferences.
    """
    for domain in ["food", "pricerange", "area"]:
        for s in ALTS[domain]:
            # If the current value is in the set find another and test it.
            if dialog_state["values"][domain] in s:
                for member in s:
                    new_prefs = copy.deepcopy(dialog_state)
                    new_prefs["values"][domain] = member
                    rs = query_restaurant_info(generate_query(new_prefs))
                    if len(rs):
                        return f"Sorry, no restaurant is found given your preferences. You can try {domain} to be {member} instead."
    # If there is nothing we can do anymore.
    return "Sorry, no restaurants found with your preferences. Try something else!"


def get_suggest_msg(dialog_state):
    """
    Returns the prompt with information about the current suggested restaurant.
    """
    r = dialog_state["suitable_restaurants"][dialog_state["current_index"]]
    return f"{r['restaurantname']} serves {r['pricerange']} priced {r['food']} food at the {r['area']} part of town."


def suggest(dialog_state):
    """
    Fills the suitable_restaurants field with suggestions, and sets
    the index to 0.
    """
    msg = None
    query = generate_query(dialog_state)
    if len(query) > 0:
        suggested = query_restaurant_info(query)
        dialog_state["suitable_restaurants"] = suggested
        dialog_state["current_index"] = 0
        if len(suggested) > 0:
            msg = get_suggest_msg(dialog_state)

    return dialog_state, msg


def inform_response(dialog_state, user_utterance):
    """
    Create response in case of inform dialog act
    :param dialog_state: The state dict.
    :param user_utterance: The user input.
    :return: dialog_state, system utterance
    """
    prefs, conf = extract_preferences(user_utterance)
    # Updating the state with w/e is extracted from the utterance.
    dialog_state["values"].update(prefs)
    dialog_state["confident"].update(conf)
    #     # If confident of some slots is false it means we used a levansthein distance ==> not sure about it,
    #     # ask to be sure. We only ask one slot at a time.
    msg = generate_question(dialog_state)
    if msg:  # Not all slots are filled, because there is still a question to ask.
        # Make a suggestion to see how many restaurants are suitable so far.
        # Only do it if there is at least one slot filled with confidence.
        if any([dialog_state["confident"][d] and dialog_state["values"][d] not in [None, False, "dontcare"] for d in
                ["food", "area", "pricerange"]]):
            dialog_state, suggestion = suggest(dialog_state)
            r = dialog_state["suitable_restaurants"]

            # No match. Offers alternatives.
            if len(r) == 0:
                msg = get_alternatives_msg(dialog_state)
            # Only one match, suggest it.
            if len(r) == 1:
                msg = suggestion
            else:
                # Undo the suggestion if it is not obvious.
                dialog_state["current_index"] = 0
                dialog_state["suitable_restaurants"] = []

    if not msg:  # All slots filled.
        # Make suggestion.
        dialog_state, msg = suggest(dialog_state)
    return dialog_state, msg


def bye_response(dialog_state, user_utterance):
    dialog_state["done"] = True
    return dialog_state, "Thank you for choosing our services, goodbye!"


def confirm_response(dialog_state, user_utterance):
    for k, v in dialog_state["confident"].items():
        if v == False:
            dialog_state["confident"][k] = True
            break

    return dialog_state, generate_question(dialog_state)


def deny_response(dialog_state, user_utterance):
    # The user denied when we asked for confirmation.
    # Delete the slot.
    for k, v in dialog_state["confident"].items():
        if v == False:
            dialog_state["confident"][k] = True
            dialog_state["values"][k] = None
            break

    return dialog_state, generate_question(dialog_state)


def reqalt_response(dialog_state, user_utterance):
    # If there are other suitable restaurants suggest on of those.
    if dialog_state["current_index"] < len(dialog_state["suitable_restaurants"]) - 1:
        dialog_state["current_index"] += 1
        return dialog_state, get_suggest_msg(dialog_state)
    # There are no other suitable restaurants.
    else:
        return dialog_state, "Unfortunately there are no more restaurants with your preferences."


def info_extracting(user_utterance):
    output = []
    regexes = {
        "postcode": ["post code"],
        "addr": ["add?ress?"],
        "phone": ["phone( number)?"],
        "food": ["(type of )?food"],
        "pricerange": ["price range"],
        "area": ["area"],
        "busy": ["busy"],
        "children": ["children"],
        "long_time": ["long time"],
        "lout": ["loud"],
        "romantic": ["romantic"]
    }
    for slot_filler, slot_regexes in regexes.items():
        for slot_regex in slot_regexes:
            match = re.search("(" + slot_regex + ")", user_utterance)
            if match:
                output.append(slot_filler)

    return output


def request_response(dialog_state, user_utterance):
    reqs = info_extracting(user_utterance)
    idx = dialog_state["current_index"]
    restaurant = dialog_state["suitable_restaurants"][idx]
    # Apply inferences.
    restaurant = apply_inference(restaurant, RULES_)
    info = []
    for d in reqs:
        try:
            info.append(f"{d} is {restaurant[d]}")
        except KeyError:
            info.append(f"{d} is unknown.")
    msg = ", ".join(info)
    return dialog_state, f"Information about {dialog_state['suitable_restaurants'][idx]['restaurantname']}: " + msg


def state_transition(dialog_state: dict, user_utterance: str):
    """
    The main dialog manager though which everything runs
    :param dialog_state: The sate dictionary.
    :param user_utterance: The user input.
    :return: The system utterance and the new state.
    """
    global MIN_LEVENSHTEIN_DISTANCE
    # First look for special commands that toggle options.
    if user_utterance == "restart":
        return copy.deepcopy(
            original_state), "Welcome to the restaurant recommendation system! Please state your preferences."
    # Example: Levensthein 5
    if user_utterance.startswith("levenshtein"):
        try:
            dist = int(user_utterance.split()[-1])
            MIN_LEVENSHTEIN_DISTANCE = dist
            return dialog_state, f"Levenshtein distance set to {dist}"
        except:
            pass
    if user_utterance == "use nn":
        dialog_state["classifier"] = "nn"
        return dialog_state, "Classifier set to be the neural network."

    if user_utterance == "use baseline":
        dialog_state["classifier"] = "baseline"
        return dialog_state, "Classifier set to be the baseline system."

    if user_utterance == "caps on":
        dialog_state["allcaps"] = True
        return dialog_state, "Caps switched on."

    if user_utterance == "caps off":
        dialog_state["allcaps"] = False
        return dialog_state, "Caps switched off."

    dialog_act = extract_dialog_act(dialog_state, user_utterance)

    # respond to the different kinds of dialog acts
    if dialog_act == "inform":
        state, msg = inform_response(dialog_state, user_utterance)
        return state, msg

    elif dialog_act == "ack":
        pass  # ack_response(dialog_state) TODO

    elif dialog_act == "affirm":
        state, msg = confirm_response(dialog_state, user_utterance)
        return state, msg

    elif dialog_act == "bye":
        return bye_response(dialog_state, user_utterance)

    elif dialog_act == "confirm":
        state, msg = confirm_response(dialog_state, user_utterance)
        return state, msg

    elif dialog_act == "deny":
        state, msg = deny_response(dialog_state, user_utterance)
        return state, msg

    elif dialog_act == "hello":
        return dialog_state, "Welcome!"

    # Same as deny, as there is no difference in most practical applications.
    elif dialog_act == "negate":
        state, msg = deny_response(dialog_state, user_utterance)
        return state, msg

    elif dialog_act == "reqalts":
        state, msg = reqalt_response(dialog_state, user_utterance)
        return state, msg

    # It does the same as reqalts in case dialog act classification goes wrong.
    elif dialog_act == "reqmore":
        state, msg = reqalt_response(dialog_state, user_utterance)
        return state, msg

    elif dialog_act == "request":
        return request_response(dialog_state, user_utterance)

    # If no states match, return a question to fill slots.
    msg = generate_question(dialog_state)
    if msg is None:
        msg = "Sorry, I did not understand you."
    return dialog_state, msg


########## --- EXPERIMENT PART --- ###################

def questionnaire():
    """
    Handles the questionnaire part. Returns a dictionary containing the questions and answers.
    """
    # List of questions.
    Q = ["How would you rate your experience on a scale of 1-5?", "Are you having a good day?"]

    form = dict()
    for q in Q:
        print(q)
        ans = input()
        form[q] = ans
    return form


def generate_random_system_msg(state) -> str:
    """
    Generates a random system utterance based on the dialog state. Used
    for fillers.
    """
    return "I prefer spanish food!"


def get_target(restaurant_info):
    """
    :param restaurant_info: The pandas dataframe, in case we want to
    check our random target for validity.

    Returns a string with the goal of the conversation. For example,
    find a chinese restaurant at the south part of town.
    """
    return "Find a chinese restaurant at the south part of town."


def get_dynamic_delay(utterance) -> int:
    """
    Generates number of seconds delay (int) after
    the provided user utterance.
    """
    length = len(utterance)
    if length < 5:
        return 0
    elif length < 15:
        return 1
    elif length < 25:
        return 2
    else:
        return 3


def chatbot(state):
    msg = "Welcome to the restaurant recommendation system!"
    user_utterance = ""
    while True:
        if state["delay"] == "static_delay":
            time.sleep(2)
        elif state["delay"] == "dynamic_delay":
            time.sleep(get_dynamic_delay(user_utterance))
        if state["allcaps"]:
            msg = msg.upper()
        print(f"(System) {msg}")
        if state["done"]:
            break
        user_utterance = input("(User) ").lower()
        # With 20% chance the system returns a random utterance, based on
        # the current state. The state remains unchanged.
        if random.random() < 0.2:
            msg = generate_random_system_msg(state)
            continue
        state, msg = state_transition(state, user_utterance)


def main():
    # First element of tuple is delay type, second is boolean is_filler.
    sessions = [("no_delay", False), ("static_delay", False), ("dynamic_delay", False), ("no_delay", True),
                ("static_delay", True), ("dynamic_delay", True)]
    random.shuffle(sessions)
    data = []
    introduction = "Introduction to the system and the experiment."
    print(introduction)
    for i, (d, f) in enumerate(sessions):
        print(f"Experiment {i + 1}/{len(sessions)}")
        print(f"The goal is: {get_target(restaurant_info)}")
        state = copy.deepcopy(original_state)
        state["delay"] = d
        state["is_filler"] = f
        chatbot(state)
        print("===================")
        print("Please fill in the questionnaire about the last conversation.")
        ans = questionnaire()
        ans["delay"] = d
        ans["is_filler"] = f
        data.append(ans)
    print("Thank you for participating in our experiment!")
    # Save the answers.
    with open(f"{int(time.time())}.json", 'w') as file:
        json.dump(data, file)


main()
