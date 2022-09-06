import argparse
import json
import os
import pickle as pk
import time
from ast import arg
from re import S

import torch
from pytorch_lightning import seed_everything
from sklearn.utils import check_random_state
from torch.nn import functional as F

from datasethandler import ClassificationReportPreprocessor, NarrationDataSet
from inferenceUtils import PerformanceNarrator
from trainer_utils import get_model

# Disable WANDB
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ClassificationPerformanceNarration:
    def __init__(self, model_base,
                 model_base_dir,
                 
                 modeltype='earlyfusion',
                 max_preamble_len=160,
                 max_len_trg=185,

                 length_penalty=8.6,
                 beam_size=10,
                 repetition_penalty=1.5,
                 return_top_beams=1,
                 lower_narrations=True,
                 process_target=True, random_state=None):
        '''
        The class handles the generation of the textual narratives explaining the implications of the classification performance of a given ML model. It requires that the path to the trained NLG model. 
        Args:
            model_base: the type of pre-trained language model (any of the following is accepted:  t5-small, t5-base, t5-large, Bart-base,      and Bart-large
            model_base_dir: the path or location of the trained model or where the trained model and its configurations were saved.
            modeltype: "baseline" or "earlyfusion" . see the article for the difference between the two types supported
            
            max_preamble_len: the number of input tokens passed to the neural generator
            max_len_trg: the number of output tokens returned neural generator

            The following parameters are used by the beamsearch algorithm 
            length_penalty, beam_size, repetition_penalty,return_top_beams


        '''
        # Check if model has been trained and all the configuration files are present
        if not (os.path.exists(model_base_dir+'/parameters.json') and os.path.exists(model_base_dir+'/trainer_state.json')):
            raise BaseException(
                "Model has not been trained. `parameters.json` and `trainer_state.json` not found")
        self.model_base = model_base
        self.model_base_dir = model_base_dir
        self.modeltype = modeltype

       
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.beam_size = beam_size
        self.return_top_beams = return_top_beams
        self.max_preamble_len = max_preamble_len
        self.max_len_trg = max_len_trg
        self.max_rate_toks = 8
        self.lower_narrations = lower_narrations
        self.process_target = process_target

        if random_state is None:
            random_state = 456

        self.seed = random_state
        self.random_state = check_random_state(random_state)
        self.setup_performed = False

    def buildModel(self, enable_sampling_inference=False, verbose=False):
        # Set up the model along with the tokenizers and other important stuff required to run the generation
        params_dict = json.load(open(self.model_base_dir+'/parameters.json'))
        #state_dict = json.load(open(args.model_base_dir+'/parameters.json'))
        best_check_point = params_dict['best_check_point']
        best_check_point_model = best_check_point + '/pytorch_model.bin'

        narrationdataset = NarrationDataSet(self.model_base,
                                            max_preamble_len=self.max_preamble_len,
                                            max_len_trg=self.max_len_trg,
                                            max_rate_toks=self.max_rate_toks,
                                            lower_narrations=self.lower_narrations,
                                            process_target=self.process_target)

        narrationdataset.build_default()

        tokenizer = tokenizer_ = narrationdataset.tokenizer_

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        performance_narration_model = get_model(narrationdataset,
                                                model_type=self.modeltype)()

        state_dict = torch.load(best_check_point_model)
        performance_narration_model.load_state_dict(state_dict)

        # This object is for processing the performance metrics provided in the  json format
        #classification_report_preprocessor = ClassificationReportPreprocessor(self.class_labels, self.is_balance)

        self.narrator = PerformanceNarrator(performance_narration_model,
                                            narrationdataset,
                                            device,
                                            classificationReportProcessor=None,
                                            sampling=enable_sampling_inference,
                                            verbose=verbose)

        self.setup_performed = True

    def generateTextualExplanation(self, prediction_report, class_labels,
                                   is_balance=True,):
        # sample of the prediction_report is {'F1-score':["20.56%","LOW"],'Accuracy':["40.16%","LOW"],'AUC':["70.16%","LOW"]}
        assert self.setup_performed, "Run the function/method ``buildModel`` before calling this method"
        assert type(prediction_report) is dict, '''The prediction report format is invalid. It should be a dictionary such as 
                                                    {'Metric_1':["Metric_1_score_value","Metric_1_score_rate"],'Metric_2':["Metric_2_score_value","Metric_2_score_rate"],...,'Metric_n':["Metric_n_score_value","Metric_n_score_rate"]} 
                                                    example is: {'F1-score':["20.56%","LOW"],'Accuracy':["40.16%","LOW"],'AUC':["70.16%","LOW"]}
                                                    '''

        if class_labels is None:
            class_labels = self.class_labels

        # This object is for processing the performance metrics provided in the  json format
        classification_report_preprocessor = ClassificationReportPreprocessor(
            class_labels, is_balance)
        self.narrator.classificationReportProcessor = classification_report_preprocessor
        generatedTexts = self.narrator.singleNarrationGeneration(prediction_report,
                                                                 self.seed,
                                                                 max_length=self.max_len_trg,
                                                                 length_penalty=self.length_penalty,
                                                                 beam_size=self.beam_size,
                                                                 repetition_penalty=self.repetition_penalty,
                                                                 return_top_beams=self.return_top_beams)

        return generatedTexts
