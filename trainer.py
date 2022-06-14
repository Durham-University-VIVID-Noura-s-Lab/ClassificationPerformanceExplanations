import argparse
import json
import os

from pytorch_lightning import seed_everything
from torch.nn import functional as F
from src.datasethandler import DataSetLoader, NarrationDataSet, train_data_original_path, train_data_permutated_path, test_data_path
#from composer import *
from src.trainer_utils import (CustomTrainerFusion, get_model,
                               getTrainingArguments, EarlyStoppingCallback)
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser(
    description='Arguments for Performance Narration Models.')
parser.add_argument('--run_id', '-run_id', type=str, default='')
parser.add_argument('-mt', '--modeltype', type=str, default='baseline', required=True,
                    help="Specifies the model type: baseline, earlyfusion")

parser.add_argument('-mb', '--modelbase', type=str, default='t5-base', required=True,
                    help="Specifies the model base: t5-base, t5-small or t5-large ")
parser.add_argument('-seed', '--seed', type=int, default=43)
parser.add_argument('--num_train_epochs', '-nb_epochs', default=20, type=int)
parser.add_argument('--evaluation_strategy',
                    '-evaluation_strategy', default="steps", )
parser.add_argument('--lr_scheduler_type', '-lr_scheduler',
                    default='cosine', type=str)
parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4)
parser.add_argument('--weight_decay', '-weight_decay', type=float, default=0.3)
parser.add_argument('-wr', '--warmup_ratio', type=float, default=0.15)
parser.add_argument('-bottom_k', '--bottom_k', type=int, default=11)
parser.add_argument('--per_device_train_batch_size',
                    '-train_bs', type=int, default=8,)

parser.add_argument('-only_eval', '--only_eval', action="store_true")
parser.add_argument('-sc', '--seed_check', action="store_true")
parser.add_argument('-use_raw', '--use_raw', action="store_true")
parser.add_argument('-sbs', '--sample_bs', action="store_true")
parser.add_argument('--output_dir', '-output_dir', type=str, required=True)
parser.add_argument('--logging_steps', '-logging_steps', default=500,)
parser.add_argument('--report_to', '-report_to', default=None, type=str)
parser.add_argument('--per_device_eval_batch_size',
                    '-eval_bs', type=int, default=4,)
parser.add_argument('-org', '--use_original_data', action="store_true",
                    help="Specifies if the training should performed using the original dataset. Default is using permuated data.")


args = parser.parse_args()
# Build the Dictionary
params_dict = vars(args)
seed_everything(args.seed)

train_arguments = {k: v for k, v in params_dict.items() if k not in ['use_raw', 'sample_bs', 'bottom_k',
                                                                     'only_eval', 'seed_check', 'modeltype',
                                                                     'iterative_gen', 'modelbase', 'inference_dir',
                                                                     'inf_sample', 'run_id',
                                                                     'max_full_len', 'use_original_data', ]}

# Set up the output directory to save the trained models

pre_trained_model_name = args.modelbase.split(
    '/')[1] if 'bart' in args.modelbase else args.modelbase
args.output_path = args.output_dir
output_path = args.output_path+'/trainednarrators/' + \
    args.modeltype + '/'+pre_trained_model_name+'/'

# When the training is performed with different random seeds
if args.seed_check:
    output_path = output_path+f'/{args.seed}/'
try:
    os.makedirs(output_path)
except:
    pass

print(f'The trained will be saved @: {output_path}')
train_arguments['output_dir'] = output_path

# Save the arguments for later (when inference is performed)


# Load the dataset based on the value of args.use_original_data
dataset = DataSetLoader(train_data_path=train_data_permutated_path, test_data_path=test_data_path) if not args.use_original_data else DataSetLoader(
    train_data_path=train_data_original_path, test_data_path=test_data_path)


# Process the data and set up the tokenizer
narrationdataset = NarrationDataSet(args.modelbase,
                                    max_preamble_len=160,
                                    max_len_trg=185,
                                    max_rate_toks=8,
                                    lower_narrations=True,
                                    process_target=True)

narrationdataset.fit(dataset.train_data, dataset.test_data)
train_dataset = narrationdataset.train_dataset
test_dataset = narrationdataset.test_dataset
tokenizer = tokenizer_ = narrationdataset.tokenizer_


val_dataset = test_dataset


train_size = int(len(train_dataset))
val_size = int(len(test_dataset))
print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))


# Build actual trainingArgument object
training_arguments = getTrainingArguments(train_arguments)


# Setup the narration model
getModel = get_model(narrationdataset, model_type=args.modeltype)

# Setup the trainer
trainer = CustomTrainerFusion(model_init=getModel,
                              args=training_arguments,
                              train_dataset=narrationdataset.train_dataset,
                              eval_dataset=narrationdataset.test_dataset,
                              callbacks=[EarlyStoppingCallback(early_stopping_patience=4)])


# train model
trainer.train()

# Save the state object from the model training
trainer.save_state()

results = trainer.evaluate()

# get the best checkpoint
best_check_point = trainer.state.best_model_checkpoint



#
params_dict['best_check_point'] = best_check_point
params_dict['output_path'] = output_path
json.dump(params_dict, open(f'{output_path}/parameters.json', 'w'))

print(
    f'Model Training Complete. Best model is at: {params_dict["best_check_point"]}')
