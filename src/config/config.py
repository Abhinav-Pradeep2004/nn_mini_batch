import os
import pathlib
import src

NUM_INPUTS = 2
NUM_LAYERS = 3
P = [NUM_INPUTS,2,1]

#f_dict = dict(zip(range(0,5),["linear","sigmoid","tanh","relu","leaky relu"]))
f = [None,"linear","sigmoid"]

LOSS_FUNCTION = "Mean Squared Error" #binary cross entropy
MINI_BATCH_SIZE = 2

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent
#pathlib provides generators to fetch the data from the directories

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")
#o/p- "src/datasetsccd "

SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_models")
#"/src/trained_models"



