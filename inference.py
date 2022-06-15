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
from src.datasethandler import DataSetLoader, NarrationDataSet, train_data_original_path, train_data_permutated_path, test_data_path
from src.inferenceUtils import PerformanceNarrator
#from composer import *
from src.trainer_utils import (CustomTrainerFusion, get_model,
                               getTrainingArguments, EarlyStoppingCallback)


os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser(
    description='Arguments for Performance Narration Models.')
parser.add_argument('--model_base_dir', '-model_base_dir',
                    type=str, required=True)
            
#parser.add_argument('-report', '--report', type=json.loads, help="""
# Expect the report to be of the format {"classes": [class_names],"is_balance": 0 (for imbalance dataset) or 1 ()}
#""")

args = parser.parse_args()
# Build the Dictionary
params_dict = vars(args)

# Check if model has been trained
if not (os.path.exists(args.model_base_dir+'/parameters.json') and os.path.exists(args.model_base_dir+'/trainer_state.json')):
    raise BaseException(
        "Model has not been trained. `parameters.json` and `trainer_state.json` not found")

params_dict = json.load(open(args.model_base_dir+'/parameters.json'))
state_dict = json.load(open(args.model_base_dir+'/parameters.json'))

best_check_point = state_dict['best_check_point']
best_check_point_model = best_check_point + '/pytorch_model.bin'

seed_everything(params_dict['seed'])

# Load the dataset based on the value of args.use_original_data
dataset_raw = DataSetLoader(train_data_path=train_data_permutated_path, test_data_path=test_data_path) if not params_dict['use_original_data'] else DataSetLoader(
    train_data_path=train_data_original_path, test_data_path=test_data_path)


# Process the data and set up the tokenizer
narrationdataset = NarrationDataSet(params_dict['modelbase'],
                                    max_preamble_len=160,
                                    max_len_trg=185,
                                    max_rate_toks=8,
                                    lower_narrations=True,
                                    process_target=True)

narrationdataset.dataset_fit(dataset_raw.test_data)
test_dataset = narrationdataset.base_dataset
tokenizer = tokenizer_ = narrationdataset.tokenizer_



device = torch.device( 'cuda') if torch.cuda.is_available() else torch.device('cpu')
# Build model
performance_narration_model = get_model(
    narrationdataset, model_type=params_dict['modeltype'])()

# Load the weights
state_dict = torch.load(best_check_point_model)
performance_narration_model.load_state_dict(state_dict)

print('Model loaded')

narrator = PerformanceNarrator(performance_narration_model,narrationdataset,device,sampling=False,verbose=False)


example = dataset_raw.test_data[99]
seed = params_dict['seed']

narratives = narrator.multipleNarrationGeneration(dataset_raw.test_data[:4], seed,  max_length=190,
                          length_penalty=8.6, beam_size=10,
                          repetition_penalty= 1.5,
                           return_top_beams=4)

print(narratives)