# Dialog System

## Quick installation

The project uses PyTorch. It can be installed by following the instructions [here](https://pytorch.org/get-started/locally/).

If you already have the neccesary packages installed, you can run the dialog system. Go to the root of the main directory and type the following command in the command line:
- python dialog_system.py

## Installation from scratch

If you need one or more packages, here is the complete list of packages:
- pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
- pip install numpy
- pip install sklearn
- pip install pandas
- pip install python-levenshtein

After installing these run the following command:
- python dialog_system.py


## Inference rules

These are the 14 rules that are implemented in the system:

```
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
Rule(14, [Antecedent("food", "chinese"),  Antecedent("food_quality", "good food")], "children", True)
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

## Experiment

The system for the experiment is located at `experiment.py`. The relevant functions are at the
bottom of the file.
* `questionnaire()` contains a list of all questions that will be asked after each interaction.
* `generate_random_system_msg(state)` can be used to generate a random message based on the
current dialog state. This message will be shown to the user during `filler` interactions.
* `get_target(restaurant_info)` generates a goal for the user, eg a location and a food type
to search for.
* `get_dynamic_delay(utterance)` returns the number of seconds delay after the user utterance
when in dynamic delay mode.

The answers to the questions will be saved in a json file.