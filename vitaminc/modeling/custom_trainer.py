
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


    """def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:

        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size
        logger = logging.get_logger(__name__)

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            print("LOGGY", logits.shape)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                print(type(logits),loss,labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            import numpy as np
            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    print("LGG ", logits)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)"""


