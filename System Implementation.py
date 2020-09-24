import pickle
import sys
import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from tensorflow.python.autograph.pyct import transformer

from tensorflow.python.keras.models import model_from_json
import pandas
#terminal: pip install python-levenshtein
from Levenshtein import distance as levenshtein_distance


# load json and create model
json_file = open('data/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("data/model.h5")


restaurant_info = pandas.read_csv("data/restaurant_info.csv")
transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("data/feature.pkl", "rb")))

# Constants
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
    "states": []
}

domain_terms = {
    "food": list(restaurant_info["food"].dropna().unique()),
    "area": list(restaurant_info["area"].dropna().unique()),
    "pricerange": list(restaurant_info["pricerange"].dropna().unique())
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

    preferences = {
        "food": "dontcare",
        "area": "dontcare",
        "pricerange": "dontcare"
    }

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
            "(?<=(a\s))(\w+)(?=\srestaurant)"
        ],
        "pricerange": [
            domain_terms["pricerange"]
        ]
    }


    for slot_filler, slot_regexes in regexes.items():
        for slot_regex in slot_regexes:
            if type(slot_regex) is list:
                for regex in slot_regex:
                    match = re.search("(" + regex + ")", dialog_state["user_utterance"])
                    if match:
                        break
            else:
                match = re.search(slot_regex, dialog_state["user_utterance"])

            if match:
                preferences[slot_filler] = match.group(1)
                break



    # Miscellaneous cases (e.g. any)
    if re.search("^any$", dialog_state["user_utterance"]):
        if ("food" in current_state):
            preferences["food"] = "dontcare"
        elif ("area" in current_state):
            preferences["area"] = "dontcare"
        elif ("price" in current_state):
            preferences["pricerange"] = "dontcare"

    for pref_name, pref_value in preferences.items():
        if pref_value == "any":
            preferences[pref_name] = "dontcare"

    return preferences




def levenshtein_edit_distance(pref_name, pref_value, domain_terms):
    """
    Use Levenshtein edit distance as implemented in the python-Levenshtein library to map values
    to the closest domain term in case an exact match is not found.
    :return:
    """
    min_distance = 3
    best_match = []

    for term in domain_terms:
        distance = levenshtein_distance(pref_value, term)
        if distance <= min_distance:
            min_distance = distance
            best_match.append({"term": term,
                               "distance": distance})

    if not best_match:
        return False
    else:
        best_match = list(filter(lambda d: d["distance"] == min_distance, best_match))
        best_match = random.shuffle(best_match)

    return best_match[0]


def rule_based_dialog_classifier(user_utterance):
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


def inform_response(dialog_state):
    """
    Create response in case if inform dialog act
    :param dialog_state:
    :return: dialog_state
    """

    system_utterance = ""
    preferences = extract_preferences(dialog_state)

    # Apply Levenshtein Edit Distance if no exact match with domain terms is found
    for pref_name, pref_value in preferences.items():
        if pref_value != "dontcare" and pref_value not in domain_terms[pref_name]:
            levenshtein = levenshtein_edit_distance(pref_name=pref_name,
                                                    pref_value=pref_value,
                                                    domain_terms=domain_terms[pref_name])

            if levenshtein == False:
                dialog_state["system_utterances"].insert(0, f'I am sorry but there is no restaurant with {pref_value} {pref_name}.')
                return dialog_state
            else:
                preferences[pref_name] = levenshtein


    # Fill slots
    for slot in dialog_state["slots"]:
        if preferences[slot["name"]] != "":
            slot["filler"] = preferences[slot["name"]]
            break

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

        restaurant_filters = random.shuffle(restaurant_filters).remove()
        query = "(" + ") and (".join(restaurant_filters) + ")"
        dialog_state["alternative_restaurants"] = query_restaurant_info(query)
        restaurant_one = dialog_state["alternative_restaurants"][0]
        restaurant_two = dialog_state["alternative_restaurants"][1]

        system_utterance = f"""there are no suggestions that satisfy your preferences.
          The following restaurants are available:
      1. Restaurant {restaurant_one["name"]} serving {restaurant_one["food"]} in {restaurant_one["area"]} part of town for {restaurant_one["price_type"]}
      2. Restaurant {restaurant_two["name"]} serving {restaurant_two["food"]} in {restaurant_two["area"]} part of town for {restaurant_two["price_type"]}
      Do you want to:
      a. change your preferences
      b. choose one of these alternatives?
      """

        dialog_state["system_utterances"].insert(system_utterance, 0)  # TODO: Improve
        dialog_state["states"].insert(0, f"Suggest restaurant")


    # Assignment: If the number of restaurants satisfying the current set of preferences is 1, then the system should not ask any remaining preferences and should immediately present the recommendation.
    elif num_matched_restaurants == 1:
        dialog_state["system_utterances"].insert("You can eat Bart's ... ", 0)  # TODO: Improve

    # Assigment: If the number of restaurants satisfying the current set of preferences is 2 or more then the system should proceed normally.
    elif num_matched_restaurants > 1:
        for slot in dialog_state["slots"]:
            if slot["filler"] == "dontcare":
                dialog_state["system_utterances"].insert(0, slot["question"])
                dialog_state["states"].insert(0, f"ask {slot['name']}")
                break

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
    dialog_act_prob = dialog_act_softmax.max()

    if dialog_act_prob < 0.8:
        dialog_act = rule_based_dialog_classifier(dialog_state["user_utterance"])
        dialog_act_prob = 1

    # respond to the different kinds of dialog acts
    if dialog_act == "inform":
        dialog_state = inform_response(dialog_state)

    elif dialog_act == "reqalt":
        pass  # reqalt_response(dialog_state) TODO

    elif dialog_act == "deny":
        pass  # deny_response(dialog_state) TODO

    elif dialog_act == "affirm":
        pass  # affirm_response(dialog_state) TODO

    elif dialog_act == "null":
        pass

    elif dialog_act == "restart":
        pass



    # TODO: Needs to be expanded

    return dialog_state


# the system-user interaction in the python console
counter = 0
try:
    while True:
        welcome_message = "System: Hello, welcome to the Cambridge restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?"
        if counter == 0:
            dialog_state["user_utterance"] = input(welcome_message).lower()
            dialog_state = state_transition(dialog_state)
            counter += 1
            continue

        # if there are no system utterances available, use welcome message
        if not dialog_state["system_utterances"]:
            user_utterance = input(welcome_message)
        else:
            user_utterance = input(f'System: {dialog_state["system_utterances"][0]}')

        dialog_state = state_transition(dialog_state)
        counter += 1

except EOFError as e:
    sys.exit("Thanks and goodbye!")