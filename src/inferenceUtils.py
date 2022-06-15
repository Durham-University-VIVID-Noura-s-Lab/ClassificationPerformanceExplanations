import copy
import functools
import re
import numpy as np
from pytorch_lightning import seed_everything
from src.datasethandler import ClassificationReportPreprocessor


# class for handling the performance narration


class PerformanceNarrator:
    def __init__(self, model,
                 experiments_dataset,
                 device,
                 classificationReportProcessor: ClassificationReportPreprocessor= None,
                 max_iter=5,
                 sampling=True,
                 verbose=False) -> None:
        self.model = model.to(device)
        self.experiments_dataset = experiments_dataset
        self.vectorizer = vectorizer = lambda x: experiments_dataset.transform(
            x)
        self.max_iter = max_iter
        self.sampling = sampling
        self.verbose = verbose
        self.device = device
        self.classificationReportProcessor = classificationReportProcessor

    def multipleNarrationGeneration(self, prediction_summaries, seed,  max_length=190,
                                    length_penalty=8.6, beam_size=10,
                                    repetition_penalty=1.5,
                                    return_top_beams=4):
        narrations = []
        for prediction_summary in prediction_summaries:
            narratives = self.singleNarrationGeneration(prediction_summary, seed,  max_length,
                                                        length_penalty, beam_size,
                                                        repetition_penalty,
                                                        return_top_beams)
            narrations.append(narratives)
        return narrations

    def singleNarrationGeneration(self, prediction_summary, seed,  max_length=190,
                                  length_penalty=8.6, beam_size=10,
                                  repetition_penalty=1.5,
                                  return_top_beams=4):
        """
        Generates the Performance Narratives by first preprocessing the performance summary report then calling the 
        __generateNarration(args) function
        """
        if self.classificationReportProcessor:
            example, placeholder_map = self.classificationReportProcessor(
            prediction_summary)
            narrative = self.generateNarration(example, seed,  max_length,
                                                length_penalty, beam_size,
                                                repetition_penalty,
                                                return_top_beams)
            narrative = [functools.reduce(lambda a, kv: a.replace(*kv), placeholder_map.items(),
                                        re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [narrative]]
        else:
            narrative = self.generateNarration(prediction_summary, seed,  max_length,
                                                length_penalty, beam_size,
                                                repetition_penalty,
                                                return_top_beams)

        return narrative[0]

    def generateNarration(self, example, seed,  max_length=190,
                            length_penalty=8.6, beam_size=10,
                            repetition_penalty=1.5,
                            return_top_beams=4):
        """
        This function is called to generate the narrative from a given preprocessed  performance summary. 
        We assume that the performance summary has been preprocessed. 
        """
        self.model.generator.eval()
        if self.model.aux_encoder is not None:
            self.model.aux_encoder.eval()
        
        seed_everything(seed)

        device = self.device

        example = copy.deepcopy(example)
        sample_too = self.sampling
        bs = beam_size

        batch = self.vectorizer(example)

        met, rate, val = batch['metrics_seq'].to(
            device), batch['rates'].to(device), batch['values'].to(device)
        clb, di = batch['class_labels'].to(
            device), batch['data_info'].to(device)
        met_att = batch['metrics_attention'].to(device)
        rate_att = batch['rate_attention'].to(device)
        val_att = batch['value_attention'].to(device)

        preamble_tokens = batch['preamble_tokens'].to(device)
        preamble_attention_mask = batch['preamble_attention_mask'].to(
            device)
        labels = batch['labels'].to(device)
        self.experiments_dataset.tokenizer = self.experiments_dataset.tokenizer_

        if self.model.aux_encoder is not None:
            table_rep = self.model.performAuxEncoding(
                [met.unsqueeze(0), met_att.unsqueeze(0)], [val.unsqueeze(0),
                                                           val_att.unsqueeze(0)], [rate.unsqueeze(0), rate_att.unsqueeze(0)])

            sample_outputs = self.model.generator.generate(input_ids=preamble_tokens.unsqueeze(0),
                                                           attention_mask=preamble_attention_mask.unsqueeze(
                                                               0),
                                                           table_inputs=table_rep,
                                                           table_attention_mask=None,
                                                           num_beams=bs,
                                                           repetition_penalty=repetition_penalty,
                                                           length_penalty=length_penalty,
                                                           early_stopping=True,
                                                           use_cache=True,
                                                           max_length=max_length,
                                                           no_repeat_ngram_size=2,
                                                           num_return_sequences=1,
                                                           do_sample=sample_too
                                                           )
        else:
            sample_outputs = self.model.generator.generate(input_ids=preamble_tokens.unsqueeze(0),
                                                           attention_mask=preamble_attention_mask.unsqueeze(
                                                               0),
                                                           num_beams=bs,
                                                           repetition_penalty=repetition_penalty,
                                                           length_penalty=length_penalty,
                                                           early_stopping=True,
                                                           use_cache=True,
                                                           max_length=max_length,
                                                           no_repeat_ngram_size=2,
                                                           num_return_sequences=1,
                                                           do_sample=sample_too
                                                           )
        sentence_output = [self.experiments_dataset.tokenizer.decode(s, skip_special_tokens=True,
                                                                     clean_up_tokenization_spaces=True) for s in sample_outputs]

        return sentence_output


class InferenceRoutine(object):
    def __init__(self) -> None:
        pass
