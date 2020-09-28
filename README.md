# INFOMAIR_group5

The project is run from the python file System Implementation.py
In line 53 there is a variable "first_run" declared which is set to True. With this variable set to True
all the following classifiers are trained, tested and saved:
- MajorityClassifier
- KeywordClassifier
- rule_based
- DecisionTreeClassifier
- NeuralNetworkClassifier
After the first run, al the models are saved and the "first_run" variable can set to False to improve 
the speed of the program for the rest of the parts.

##Interactive Part\
The interactive part can be turned off by setting the variable 
"interactive_part" to FALSE. Initially it is set to TRUE. It this case there is asked for an utterance, 
until the user enters 'STOP'. Until this is done the program keeps asking for new utterances over and over again.

##Dialog Manager\
If the interactive part is finished, the program starts the dialog manager. The program starts with a welcome 
messenger and asks for a user input. Thereafter, every system utterance is expecting an user input. 
