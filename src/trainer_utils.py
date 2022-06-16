import torch
from src.modeling_bart import BartNarrationModel
from src.modeling_t5 import T5NarrationModel
from transformers import TrainingArguments, Trainer


class CustomTrainerFusion(Trainer):
    vocab_size: int
    scale_loss: bool = False
    prev_seq_loss = []
    ce_loss = []
    total_loss = []
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    def compute_loss(self, model, inputs, return_outputs=False):
        comp_input = inputs
        outputs = model(comp_input,)

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def getTrainingArguments(arg_dict):
    return TrainingArguments(
        **arg_dict,
        overwrite_output_dir=True,
        adafactor=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        disable_tqdm=True,
    )


def get_model(dataset, model_type):
    def getModel():
        if 'bart' in dataset.modelbase:
            return BartNarrationModel(
                vocab_size=len(dataset.tokenizer_), model_type=model_type,
                modelbase=dataset.modelbase,)
        else:
            return T5NarrationModel(
                vocab_size=len(dataset.tokenizer_), model_type=model_type,
                modelbase=dataset.modelbase,)
    return getModel
