import argparse
from ast import arg
import pickle as pk
import time
from re import S

from pytorch_lightning import seed_everything
from torch.nn import functional as F

from composer import *
from src.losses import computeLoss
from src.model_utils import setupTokenizer
from src.narrations_models import BartNarrationModel
from src.trainer_utils import (CustomTrainerFusion, get_bart_model,
                               getTrainingArguments,EarlyStoppingCallback)

processed = pk.load(open('dataset/train_dataset_new.dat', 'rb'))

# Load the test set
test_data = json.load(open('dataset/test set.json'))
test_sample = []
eval_tables = []
for pc in test_data:
    test_sample.append(processInputTableAndNarrations(
        pc, identical_metrics=identicals))


rtest_sample = []
reval_tables = []
for pc in test_data:
    rtest_sample.append(processInputTableAndNarrations(
        pc, identical_metrics=identicals))


# Process the data and set up the tokenizer
narrationdataset = NarrationDataSet(args.modelbase,
                                    max_preamble_len=160,
                                    max_len_trg=185, max_rate_toks=8,
                                    lower_narrations=True,
                                    process_target=True)

narrationdataset.fit(processed, test_sample)

dataset = narrationdataset.train_dataset
test_dataset = narrationdataset.test_dataset
tokenizer = tokenizer_ = narrationdataset.tokenizer_


train_dataset, val_dataset = dataset, test_dataset


train_size = int(len(dataset))
val_size = int(len(test_dataset))
print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))


# Build actual trainingArgument object
training_arguments = getTrainingArguments(train_arguments)

if 'bart' in args.modelbase:
    getModel = get_model(narrationdataset,
                          model_type=args.modeltype)

# Setup the trainer
trainer = CustomTrainerFusion(model_init=getModel,
                              args=training_arguments,
                              train_dataset=narrationdataset.train_dataset,
                              eval_dataset=narrationdataset.test_dataset,
                              callbacks=[EarlyStoppingCallback(early_stopping_patience=4)])


# train model
trainer.train()



