import random
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.nn import Module, ReLU, Linear, Sequential, Dropout

import copy
import pandas
# terminal: pip install python-levenshtein
from Levenshtein import distance as levenshtein_distance
from collections import defaultdict
import time
import json
import warnings

# remove warnings for a better user experience
warnings.filterwarnings("ignore")


class Antecedent:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"[{self.name, self.value}]"

    def __str__(self):
        return self.__repr__()


class Rule:
    def __init__(self, identifier, antecedents, consequent, truth_value):
        self.identifier = identifier
        self.antecedents = antecedents
        self.consequent = consequent
        self.truth_value = truth_value
        self.iteration = None

    def __repr__(self):
        return f"{self.identifier}. {self.antecedents} => ({self.consequent}, {self.truth_value})"

    def __str__(self):
        return self.__repr__()


RULES_ = [
    Rule(1, [Antecedent("pricerange", "cheap"), Antecedent("food_quality", "good food")], "busy", True),
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
    Rule(12, [Antecedent("food", "french")], "romantic", True),
    Rule(13, [Antecedent("food", "italian")], "romantic", True),
    Rule(14, [Antecedent("food", "chinese"), Antecedent("food_quality", "good food")], "children", True)
]


def is_applicable(rule, restaurant):
    """
    Predicate function, returns whether a rule is applicable
    to the given restaurant.
    """
    fields = restaurant.keys()
    applicable = False

    for a in rule.antecedents:

        # first check if antecedent is true, then check is consequent is not already present in the restaurant info
        if (a.name in restaurant.keys() and a.value == restaurant[a.name]) and \
                rule not in restaurant["applied_rules"]:
            applicable = True

        else:
            applicable = False
            break

    return applicable


def apply_inference(restaurant, rules, dialog_state):
    """
    Applies the provided rules to the restaurant instance.
    :param restaurant: A dictionary representing a restaurant.
    :param rules: The set of rules to be applied.
    :return: The restaurant dictionary with additional fields that
    are the result of the implications.
    """
    applicable = [...]
    iteration = 0
    msg = ""

    restaurant["applied_rules"] = []

    while applicable:
        iteration += 1
        applicable = [r for r in rules if is_applicable(r, restaurant)]

        for rule in applicable:
            restaurant[rule.consequent] = rule.truth_value
            rule.iteration = iteration
            restaurant["applied_rules"].append(rule)

            antecedents = [a.name + ", " + str(a.value) for a in rule.antecedents]
            antecedents_str = "[" + ("], [").join(antecedents) + "]"

            msg += f"Iteration: {iteration}. Rule {rule.identifier}. {antecedents_str} > {rule.consequent} = {rule.truth_value}\n"

    msg += check_preferences_with_rules(restaurant, dialog_state)

    return msg


def check_preferences_with_rules(restaurant, dialog_state):
    """
    This functions checks whether the additional preferences states by the user (e.g. romantic, long time, loud
    busy and children) are compatible with the rules
    """

    msg = ""
    a = restaurant
    rule_applied = False
    recommendations = []

    for pref_name, pref_value in dialog_state["additional_preferences"].items():
        # If the user did not state a preference, we skip it
        if pref_value == None:
            continue

        for rule in restaurant["applied_rules"]:
            if rule.consequent == pref_name:
                rule_applied = True

                antecedents = [a.name + ", " + str(a.value) for a in rule.antecedents]
                antecedents_str = "[" + ("], [").join(antecedents) + "]"

                if rule.truth_value == pref_value:
                    recommendations.append(True)
                    msg += f"From iteration: {rule.iteration}. this restaurant is recommended because of rule {antecedents_str} > {rule.consequent} = {rule.truth_value}\n"
                else:
                    msg += f"From iteration: {rule.iteration}. this restaurant is not recommended because of rule {antecedents_str} > {rule.consequent} = {rule.truth_value}\n"
                    recommendations.append(False)

    if rule_applied == False:
        msg += f"{a['restaurantname'].capitalize()} serves {a['pricerange']} priced {a['food']} food at the {a['area']} part of town.\n"

    # We use the firstly applied rule as the basis for our recommendation
    # If the rule goes against the user's wishes, we restart the system and the state
    if recommendations and recommendations[0] == False:
        dialog_state = copy.deepcopy(original_state)
        msg += "There is a conflict between our inference rule and your preference. The system will restart..\n"
        msg += f"\nWelcome to the restaurant recommendation system! Please state your preferences.\n"

    return msg


# The database
restaurant_info = pandas.read_csv("data/restaurant_info.csv", index_col=0)

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
    "additional_preferences": {
        "busy": None,
        "long_time": None,
        "children": None,
        "romantic": None,
        "loud": None,
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


def query_restaurant_info(query):
    """
    Query restaurant_info database
    :return: list of dicts representing found records in database
    """
    results = restaurant_info.query(query).to_dict('records')
    return results

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
    alternatives = []

    for domain in ["food", "pricerange", "area"]:
        for s in ALTS[domain]:
            # If the current value is in the set find another to query the database with
            if dialog_state["values"][domain] in s:
                for member in s:
                    new_prefs = copy.deepcopy(dialog_state)
                    new_prefs["values"][domain] = member.lower()
                    rs = query_restaurant_info(generate_query(new_prefs))
                    if rs:
                        for r in rs:
                            if r not in alternatives:
                                alternatives.append(r)

    number_of_alternatives = 2

    # Supplement the alternatives when needed
    for _ in range(len(alternatives), number_of_alternatives):
        random_choice = random.choice(query_restaurant_info("ilevel_0 in ilevel_0"))
        alternatives.append(random_choice)

    # Randomly pick two alternatives from the list
    alternatives_suggest = []
    for alternative in range(number_of_alternatives):
        random_choice = random.choice(alternatives)
        alternatives_suggest.append(random_choice)
        dialog_state["suitable_restaurants"].append(random_choice)
        alternatives.remove(random_choice)

    # Build up the string to offer alternatives
    alternatives_str = ""
    for i, a in enumerate(alternatives_suggest):
        alternatives_str += f"{i + 1}. {a['restaurantname'].capitalize()} serves {a['pricerange']} priced {a['food']} food at the {a['area']} part of town.\n"

    msg = f"""
There are no suggestions that satisfy your preferences.
The following suggestions are available:
{alternatives_str}
Do you want to:
a. Change your preferences (type "change")
b. Choose one of these alternatives (type "restaurant [number])"
    """

    return dialog_state, msg


def get_suggest_msg(dialog_state):
    """
    Returns the prompt with information about the current suggested restaurant.
    """
    msg = ""
    r = dialog_state["suitable_restaurants"][dialog_state["current_index"]]

    additional_preferences = any([pref != None for _, pref in dialog_state["additional_preferences"].items()])

    if additional_preferences:
        msg += apply_inference(dialog_state["suitable_restaurants"][dialog_state["current_index"]], RULES_,
                               dialog_state)
    else:
        msg += f"""{r['restaurantname'].capitalize()} serves {r['pricerange']} priced {r['food']} food at the {r['area']} part of town."""

    return msg


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
        else:
            _, msg = get_alternatives_msg(dialog_state)

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
    #     # If confident of some slots is false it means we used a levensthein distance ==> not sure about it,
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
                dialog_state, msg = get_alternatives_msg(dialog_state)
            # Only one match, suggest it.
            elif len(r) == 1:
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
        "loud": ["loud"],
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
    # restaurant = apply_inference(restaurant, RULES_)
    info = []
    for d in reqs:
        try:
            info.append(f"{d} is {restaurant[d]}")
        except KeyError:
            info.append(f"{d} is unknown.")
    msg = ", ".join(info)
    return dialog_state, f"Information about {dialog_state['suitable_restaurants'][idx]['restaurantname']}: " + msg


def difficult_cases(dialog_act, user_utterance):
    """
    Deals with difficult cases which are hard to classify
    """

    regexes = {
        "inform": [
            "looking for unintelligible"
        ],
        "deny": [
            "dont",
            "don't"
        ],
        "request": [
            "romantic",
            "busy",
            "long time",
            "children",
            "loud"
        ]
    }

    for act, regexes in regexes.items():
        for regex in regexes:
            if re.search(regex, user_utterance):
                dialog_act = act
                break

    return dialog_act


def update_additional_preferences(matches, dialog_state):
    if matches:
        for match in matches:
            for i, group in enumerate(match):
                if group != "":
                    dialog_state["additional_preferences"]["_".join(group.split())] = match[i + 1] == "true"
                    break

    return dialog_state


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

    if user_utterance == "change":
        dialog_state["suitable_restaurants"] = []
        return dialog_state, "Sure, you can now change your preferences. You can search for restaurants by area, price range or food type"

    if re.search("^restaurant (\d$)", user_utterance):
        index = int(re.search("^restaurant (\d$)", user_utterance).group(1)) - 1
        dialog_state["current_index"] = index
        return dialog_state, get_suggest_msg(dialog_state)

    regex = "(loud)\s(\w+)|(busy)\s(\w+)|(long time)\s(\w+)|(children)\s(\w+)|(romantic)\s(\w+)"
    if re.findall(regex, user_utterance):
        matches = re.findall(regex, user_utterance)
        dialog_state = update_additional_preferences(matches, dialog_state)

        return dialog_state, get_suggest_msg(dialog_state)

    dialog_act = extract_dialog_act(dialog_state, user_utterance)
    dialog_act = difficult_cases(user_utterance=user_utterance, dialog_act=dialog_act)

    # respond to the different kinds of dialog acts
    if dialog_act == "inform":
        state, msg = inform_response(dialog_state, user_utterance)
        return state, msg

    elif dialog_act == "ack":
        pass

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


#### EXPERIMENT ####

def questionnaire(Q):
    """
    Prompts the user with questions, and returns a dict with {question: answer} for
    each question, answer pair in the form.
    """
    form = dict()
    for q in Q:
        print(q)
        ans = input()
        form[q] = ans

    return form


def get_target(restaurant_info):
    """
    :param restaurant_info: The pandas dataframe containing the restaurants.
    This function generates a target for the user, for example "find the telephone number of an Italian restaurant situated in the centre".
    """
    return "Find a chinese restaurant in the south part of town."


def get_dynamic_delay(user_utterance) -> int:
    """
    This function returns the delay in seconds after the user utterance privided.
    """
    length = len(user_utterance)
    if length < 5:
        return 0
    elif length < 20:
        return 1
    else:
        return 3


def get_random_msg(state):
    """
    :param state: The dialog state.
    This function generates a string message based on the state. This message
    is used as a "random" system utterance to confuse the victim. The state will
    be left unchanged.
    """
    return "I prefer spanish food myself."


def system(state):
    """
    A single run of the chatbot.
    """
    # This logs all the utterances by the user and the system.
    utterance_log = []

    def print_(s):
        """
        Override that prints and also logs.
        """
        utterance_log.append(s)
        print(s)

    def input_(s):
        """
        Override that takes input and also logs.
        """
        i = input(s)
        utterance_log.append(f"{s} {i}")
        return i

    msg = "Welcome to the restaurant recommendation system!"
    user_utterance = ""
    total_delay = 0
    n_delays = 0
    while True:
        if state["delay"] == "static_delay":
            time.sleep(2)
        elif state["delay"] == "dynamic_delay":
            delay_delta = get_dynamic_delay(user_utterance)
            time.sleep(delay_delta)
            total_delay += delay_delta
            n_delays += 1
        print_(f"(System) {msg}")
        if state["done"]:
            break
        user_utterance = input_("(User) ").lower()
        # If it is a filler, with 20% chance give a random output and leave
        # the state unchanged.
        if state["is_filler"] and random.random() < 0.2:
            msg = get_random_msg(state)
            continue
        state, msg = state_transition(state, user_utterance)
    return utterance_log, {"total_delay": total_delay, "number_of_delays": n_delays}


def main():
    # Load the texts.
    with open("data/sys_text.json") as file:
        sys_text = json.load(file)

    # Randomize the order of dialog types, and the associated goals.
    goals = sys_text["targets"]
    random.shuffle(goals)

    sessions = [("no_delay", False), ("no_delay", False), ("dynamic_delay", False), ("dynamic_delay", False)]
    random.shuffle(sessions)

    # The log of the experiment.
    experiment_info = {}

    # ======== The actual conversation starts here.

    # Get language preference.
    lang = ""
    while lang != "en" and lang != "nl":
        lang = input("Type 'en' for english or 'nl' for dutch.")

    texts = sys_text[lang]
    # Ask for personal details.
    age = input(texts['age'])
    gender = input(texts['gender'])

    experiment_info["personal_info"] = {"age": age, "gender": gender}

    # Show the instructions.
    input(texts['introduction'])
    input(texts['tutorial'])
    input(texts['tutorial'])
    input(texts['experiment_overview'])

    # The practice dialog.
    state = copy.deepcopy(original_state)
    # Set the appropriate delay.
    state["delay"] = "no_delay"
    state["is_filler"] = False
    system(state)

    # List of dialog logs.
    dialogs = []

    for i, (d, f) in enumerate(sessions):
        print("==================================================")
        print(f"Starting experiment {i + 1}/{len(sessions)}")
        goal = goals[i]
        print(f"Goal: {goal}")
        state = copy.deepcopy(original_state)
        # Set the appropriate delay.
        state["delay"] = d
        state["is_filler"] = f

        # Run the session.
        start_t = time.time()
        utterances, delay_info = system(state)
        end_t = time.time()

        # Evaluation form.
        print("++++++++++++++++++++++++++++++++++++++++++++++++++")
        evaluation = questionnaire(texts["questions"])
        dialog_info = {"log": utterances, "duration_sec": int(end_t - start_t), "questionnaire": evaluation,
                       "delay_type": d}
        dialog_info.update(delay_info)
        dialogs.append(dialog_info)
    experiment_info["dialogs"] = dialogs

    # Save the questions and answers.
    print("Thank you for participating in the experiment!")
    with open(f"{int(time.time())}.json", 'w') as file:
        json.dump(experiment_info, file)


main()