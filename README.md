# INFOMAIR_group5

## Installation

The project uses PyTorch. It can be installed by following the instructions [here](https://pytorch.org/get-started/locally/).

After installing PyTorch you will need the Python Levenshtein library via the following command:
- pip install python-levenshtein

## Inference rules

These are the 12 rules that are implemented in the system:

```
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
```
The valid attributes that can be checked are the following: `busy, long_time, children, romantic, popular, loud`. If rules don't apply, the attribute gets the
unknown status, as it is neither justified nor overruled.

## Configuration

The following options are supported, with examples:

* Restart dialog. `restart`
* Set minimum Levenshtein edit distance. `Levenshtein 5` (Sets it to 5).
* Choose which classifier to use (baseline or neural network) `use nn` or `use baseline`.
* Introduce a delay before showing system responses. `delay on` or `delay off`.
* OUTPUT IN ALL CAPS OR NOT. `caps on` or `caps off`.
