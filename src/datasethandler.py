import json
import random

from attr import dataclass
import torch

from data_utils import cleanRatingPreamble, getClassLabels, processInputTableAndNarrations, identicals
from src.model_utils import setupTokenizer

# Change these to match the location of your dataset
train_data_permutated_path = "dataset/train_data_permutated.json"
train_data_original_path = "dataset/train_data_original.json"
test_data_path = 'dataset/test set.json'


class DataSetLoader(object):
    def __init__(self, train_data_path=train_data_permutated_path,
                 test_data_path=test_data_path) -> None:
        super().__init__()

        # By default, the permutated training dataset will be loaded
        self.train_data = json.load(open(train_data_path))
        self.test_data_raw = json.load(open(test_data_path))

        self.test_data = []
        for pc in self.test_data_raw:
            self.test_data.append(processInputTableAndNarrations(
                pc, identical_metrics=identicals))


class RDFDataSetForTableStructured(torch.utils.data.Dataset):
    def __init__(self, tokenizer,
                 data_pack,
                 modelbase,
                 nb_metrics=6,
                 max_preamble_len=150,
                 max_len_trg=200,
                 max_metric_toks=8,
                 max_val_toks=8,
                 max_rate_toks=8,
                 nb_classes=8,
                 lower_narrations=False,
                 process_target=False,
                 use_raw=False):
        super().__init__()
        self.modelbase = modelbase
        self.nb_metrics = nb_metrics
        self.tokenizer = tokenizer
        self.data_pack = data_pack
        self.max_preamble_len = max_preamble_len
        self.max_metric_toks = max_metric_toks
        self.max_val_toks = max_val_toks
        self.max_rate_toks = max_rate_toks
        self.max_len_trg = max_len_trg
        self.nb_classes = nb_classes
        self.lower_narrations = lower_narrations
        self.process_target = process_target
        self.use_raw = use_raw
        self.pad_token_id = tokenizer.pad_token_id
        self.preamble_tokenizer = lambda x: self.tokenizer(x, return_attention_mask=True,
                                                           max_length=self.max_preamble_len,
                                                           padding='max_length',
                                                           add_special_tokens=True,
                                                           truncation=True,
                                                           return_tensors='pt')

        self.metrics_tokenizer = lambda x: self.tokenizer(x, return_attention_mask=True,
                                                          max_length=self.max_metric_toks,
                                                          padding='max_length',
                                                          add_special_tokens=True,
                                                          truncation=True,
                                                          return_tensors='pt')
        self.value_tokenizer = lambda x: self.tokenizer(x, return_attention_mask=True,
                                                        max_length=self.max_val_toks,
                                                        padding='max_length',
                                                        add_special_tokens=True,
                                                        truncation=True,
                                                        return_tensors='pt')
        self.rate_tokenizer = lambda x: self.tokenizer(x, return_attention_mask=True,
                                                       max_length=self.max_rate_toks,
                                                       truncation=True,
                                                       padding='max_length',
                                                       add_special_tokens=True,
                                                       return_tensors='pt')

        self.clb_tokenizer = lambda x: self.tokenizer(x, return_attention_mask=False,
                                                      max_length=1,
                                                      truncation=True,
                                                      padding='max_length',
                                                      add_special_tokens=False,
                                                      return_tensors='pt')
        self.di_tokenizer = lambda x: self.tokenizer(x,
                                                     return_attention_mask=False,
                                                     max_length=1,
                                                     truncation=True,
                                                     padding='max_length',
                                                     add_special_tokens=False,
                                                     return_tensors='pt')

    def __len__(self,):
        return len(self.data_pack)

    def processTableInfo(self, data_row):
        data_di = data_row['dataset_attribute']
        data_clb = data_row['classes']
        data_target = data_row['narration']
        if not self.use_raw:
            data_preamble = data_row['preamble']
            data_rates = [r.strip() for r in data_row['rates']]
        else:
            data_preamble = cleanRatingPreamble(data_row['preamble'])
            data_rates = ['VALUE']*len([r.strip() for r in data_row['rates']])
        data_values = [v.strip() for v in data_row['values']]
        data_metrics = [m.strip() for m in data_row['metrics']]
        target_encoding = self.tokenizer(data_target, max_length=self.max_len_trg,
                                         padding='max_length',
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors='pt'
                                         )
        labels = target_encoding['input_ids']

        # process the class labels
        class_labels = self.clb_tokenizer(data_clb)['input_ids']
        nb_classes = class_labels.shape[0]
        if nb_classes < self.nb_classes:
            pads = torch.zeros((self.nb_classes-nb_classes, 1))
            class_labels = torch.cat(
                [class_labels, pads], dim=0).type(torch.IntTensor)

        # process the dataset information
        data_info = self.di_tokenizer(data_di)['input_ids']

        # process the dataset_info row
        values = self.value_tokenizer(data_values)

        # process the labels row
        rates = self.rate_tokenizer(data_rates)

        # process all the rows on metrics
        metrics = self.metrics_tokenizer(data_metrics)

        metrics_seq = metrics['input_ids']
        metrics_attention = metrics['attention_mask']
        nb_metrics = metrics_seq.shape[0]
        if nb_metrics < self.nb_metrics:
            pads = torch.zeros(
                (self.nb_metrics-nb_metrics, self.max_metric_toks))
            metrics_seq = torch.cat(
                [metrics_seq, pads+self.pad_token_id], dim=0).type(torch.IntTensor)
            metrics_attention = torch.cat(
                [metrics_attention, pads], dim=0).type(torch.IntTensor)

            val_pad = torch.zeros(
                (self.nb_metrics-nb_metrics, self.max_val_toks))
            rate_pad = torch.zeros(
                (self.nb_metrics-nb_metrics, self.max_rate_toks))

            values['input_ids'] = torch.cat(
                [values['input_ids'], val_pad+self.pad_token_id], dim=0).type(torch.IntTensor)
            rates['input_ids'] = torch.cat(
                [rates['input_ids'], rate_pad + self.pad_token_id], dim=0).type(torch.IntTensor)

            values['attention_mask'] = torch.cat(
                [values['attention_mask'], val_pad], dim=0).type(torch.IntTensor)
            rates['attention_mask'] = torch.cat(
                [rates['attention_mask'], rate_pad], dim=0).type(torch.IntTensor)

        preamble_encoding = self.preamble_tokenizer(data_preamble)
        preamble_tokens = preamble_encoding['input_ids']
        preamble_attention_mask = preamble_encoding['attention_mask']
        labels[labels == self.tokenizer.pad_token_id] = -100
        return dict(
            preamble_tokens=preamble_tokens.flatten(),
            preamble_attention_mask=preamble_attention_mask.flatten(),
            class_labels=class_labels.flatten(),
            data_info=data_info.flatten(),
            metrics_seq=metrics_seq.flatten(),
            metrics_attention=metrics_attention.flatten(),
            values=values['input_ids'].flatten(),
            rates=rates['input_ids'].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_encoding['attention_mask'].flatten(),
            rate_attention=rates['attention_mask'].flatten(),
            value_attention=values['attention_mask'].flatten()
        )

    def __getitem__(self, idx):
        # print(self.data_pack)
        data_row = self.data_pack[idx]
        return self.processTableInfo(data_row)


class NarrationDataSet:
    def __init__(self, modelbase, max_preamble_len=160,
                 max_len_trg=185,
                 max_rate_toks=8,
                 lower_narrations=True,
                 process_target=True,) -> None:

        # Get the tokenizer
        self.tokenizer_ = setupTokenizer(modelbase=modelbase)
        self.modelbase = modelbase
        self.max_preamble_len = max_preamble_len
        self.max_len_trg = max_len_trg
        self.max_rate_toks = max_rate_toks
        self.lower_narrations = lower_narrations
        self.process_target = process_target

    def dataset_fit(self, dataset):
        self.base_dataset = RDFDataSetForTableStructured(self.tokenizer_,
                                                         dataset,
                                                         self.modelbase, max_preamble_len=self.max_preamble_len,
                                                         max_len_trg=self.max_len_trg,
                                                         max_rate_toks=self.max_rate_toks,
                                                         lower_narrations=self.lower_narrations,
                                                         process_target=self.process_target,
                                                         use_raw=False)

    def fit(self, trainset, testset):
        self.train_dataset = RDFDataSetForTableStructured(self.tokenizer_,
                                                          trainset,
                                                          self.modelbase, max_preamble_len=self.max_preamble_len,
                                                          max_len_trg=self.max_len_trg,
                                                          max_rate_toks=self.max_rate_toks,
                                                          lower_narrations=self.lower_narrations,
                                                          process_target=self.process_target,
                                                          use_raw=False)

        self.base_dataset = self.test_dataset = RDFDataSetForTableStructured(self.tokenizer_,
                                                                             testset,
                                                                             self.modelbase, max_preamble_len=self.max_preamble_len,
                                                                             max_len_trg=self.max_len_trg,
                                                                             max_rate_toks=self.max_rate_toks,
                                                                             lower_narrations=self.lower_narrations,
                                                                             process_target=self.process_target,
                                                                             use_raw=False)

    def transform(self, pack):
        return self.base_dataset.processTableInfo(pack)


class ClassificationReportPreprocessor(object):
    def __call__(self, prediction_summary):
        '''
        prediction_summary is a dictionary with the metric names as the keys and the list [score, rate] as the values 
        '''
        preamble = "<MetricsInfo> "

        # Replace the F{beta}-score metric name with a
        fnames = {'F1-score': 'F1score', 'F1-Score': 'F1score',
                  'F2-score': 'F2score', 'F2-Score': 'F2score'}
        report = []
        m_list = []
        v_list = []
        r_list = []
        metric_maps = {}
        for metric_name, score_summary in prediction_summary.items():
            m = metric_name
            mx = m.lower().replace('-score', '').strip()
            mx = m.lower().replace(' score', '').strip()
            mx = m.lower().replace('score', '').strip()

            m = fnames.get(m, m)
            metric_maps[m.lower()] = metric_name

            score = score_summary[0]
            rate = score_summary[-1].upper()

            if '%' not in score:
                score = score+'%'
            metric_string = f'{m.lower()} | VALUE_{rate} | {score} '
            m_list.append(m.replace('-', '').replace('_', '').lower())
            v_list.append(score)
            r_list.append(rate)
            if mx.lower() in identicals.keys():
                metric_string += ' && ' + \
                    f'{m.lower()} | also_known_as | {identicals[metric_name]}'
            report.append(metric_string)

        # Get different narrative preamble
        random.shuffle(report)
        report = ' && '.join(report)+' '

        metric_maps.update(self.class_maps)
        output = {'preamble': preamble + report + self.task_section,
                  'metrics': m_list,
                  'values': v_list,
                  'rates': r_list,

                  **self.task_dictionary,
                  'narration': "",


                  }

        return output, metric_maps

    def __init__(self, classes, is_balance):
        super().__init__()
        assert len(classes) > 1, "The number of classes should be greater than 1"
        self.classes = classes

        self.class_labels_placeholders = getClassLabels(len(self.classes))
        self.class_maps = {p: c for p, c in zip(
            self.class_labels_placeholders, classes)}
        classes_string = ', '.join(
            self.class_labels_placeholders[:-1])+' and '+self.class_labels_placeholders[-1]
        is_balance = "is_balanced" if is_balance else "is_imbalanced"
        self.task_section = f"<|section-sep|> <TaskDec> ml_task | data_dist | {is_balance} && ml_task | class_labels | {classes_string}  <|section-sep|> <|table2text|> "

        self.task_dictionary = {'dataset_attribute': [is_balance],
                                'classes': self.class_labels_placeholders}
