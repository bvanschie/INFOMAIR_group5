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
        print(f"applicable in iteration: {applicable}")
        for rule in applicable:
            restaurant[rule.consequent] = rule.truth_value
    return restaurant