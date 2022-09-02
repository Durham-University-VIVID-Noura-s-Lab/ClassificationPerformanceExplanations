import argparse
from ast import arg
import json
import os
import pickle as pk
import time
from re import S

import torch

from pytorch_lightning import seed_everything
from torch.nn import functional as F
from src.datasethandler import ClassificationReportPreprocessor, DataSetLoader, NarrationDataSet, train_data_original_path, train_data_permutated_path, test_data_path
from src.inferenceUtils import PerformanceNarrator
#from composer import *
from src.trainer_utils import (CustomTrainerFusion, get_model,
                               getTrainingArguments, EarlyStoppingCallback)


os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ClassificationNarration:
  def __init__(self,model_base_dir):
    
    # Check if model has been trained and all the configuration files are present
    if not (os.path.exists(args.model_base_dir+'/parameters.json') and os.path.exists(args.model_base_dir+'/trainer_state.json')):
        raise BaseException("Model has not been trained. `parameters.json` and `trainer_state.json` not found")
    
    
    
