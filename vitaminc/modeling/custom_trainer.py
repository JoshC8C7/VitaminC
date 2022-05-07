
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.data.processors.utils import InputExample, InputFeatures
import vitaminc.modeling.clf_distill_loss_functions as losses
from typing import List, Optional, Union
from dataclasses import dataclass
from torch import exp, log
import torch
import gc

from transformers.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_numpify, IterableDatasetShard, \
    nested_truncate
from transformers.trainer_utils import EvalLoopOutput, has_length, EvalPrediction, denumpify_detensorize
from transformers.utils import logging


@dataclass
class NewInputExample(InputExample):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    bias: Optional[tuple] = None
    teach: Optional[tuple] = None

@dataclass
class NewInputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in `[0, 1]`: Usually `1` for tokens that are NOT MASKED, `0` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    bias: Optional[tuple] = (5,5,5)
    teach: Optional[tuple] = (6,6,6)


LOSS_DICT = {
"plain":losses.Plain,
"PoE":losses.BiasProductByTeacher,
"PoE_anneal":losses.BiasProductByTeacherAnnealed,
"reweight":losses.ReweightByTeacher,
"reweight_anneal":losses.ReweightByTeacherAnnealed,
"conf_reg":losses.SmoothedDistillLoss,
"conf_reg_anneal":losses.SmoothedDistillLossAnnealed,
}


class NewTrainer(Trainer):

    def __init__(self, loss_fn_name, *args, **kwargs):
        self.loss_fn = self.set_loss_fn(loss_fn_name)
        super().__init__(*args, **kwargs)

    def set_loss_fn(self,loss_name):
        loss = LOSS_DICT.get(loss_name, None)
        if loss is None and loss_name is not None:
            print("No entry found for " + loss_name + " . Choose one of" + ", ".join(LOSS_DICT.keys()))
            raise NotImplementedError
        elif loss:
            return loss()
        else:
            return losses.Plain()


    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        #Pop bias instead of get, as the model function doesn't want to receive bias.
        if 'bias' in inputs:
            biases = inputs.pop("bias")
        else:
            biases = None


        #Need to log the bias; the loss functions from Utama et al. expect the biases to be input logged, but not the teacher probs.
        if str(self.loss_fn) in ["SmoothedDistillLoss()", "SmoothedDistillLossAnnealed()"]:
            try:
                teaches = inputs.pop("teach")
            except:
                raise FileExistsError('use_teach must be set for this loss function, and a teacher file provided at e.g. data/teaches/train_alb_base.json')
            #Here, biases are from the shallow model (stored on disk unlogged), and teacher_probs from a run on albert_base (stored on disk as logs)
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = self.loss_fn.forward(hidden=None, logits=logits, bias=log(biases), teacher_probs=exp(teaches),
                                        labels=labels)
        else:
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            #Biases are passed as teach_probs purposefully here;
            # its because the loss functions are the same (notwithstanding the logging, discussed above). However,
            # annealed methods exist for the teacher_probs ones, but not the bias ones, so is easier to do this.
            loss = self.loss_fn.forward(hidden=None,logits=logits,bias=None,teacher_probs=biases,labels=labels)


        return (loss, outputs) if return_outputs else loss
