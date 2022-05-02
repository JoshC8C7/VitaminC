import os

import torch
from transformers import DataProcessor
import json
from vitaminc.processing.utils import read_jsonlines

class VitCProcessor(DataProcessor):
    def get_examples_from_file(self, file_path, set_type="train",bias_lines=None, teach_lines=None):
        k =  self._create_examples(
            read_jsonlines(file_path), set_type, bias_lines, teach_lines)
        return k

    def get_train_examples(self, data_dir,bias_dir=None, teach_dir = None):
        """See base class."""
        if bias_dir:
            with open(bias_dir,'r') as fp:
                bias_lines = json.load(fp)
        else: bias_lines = None
        if teach_dir:
            with open(teach_dir,'r') as fp:
                teach_lines = json.load(fp)
        else: teach_lines = None
        return self.get_examples_from_file(
            os.path.join(data_dir, "train.jsonl"), "train", bias_lines,teach_lines)

    def get_dev_examples(self, data_dir,bias_dir=None, teach_dir = None):
        """See base class."""
        return self.get_examples_from_file(
            os.path.join(data_dir, "dev.jsonl"), "dev")

    def get_test_examples(self, data_dir,bias_dir=None, teach_dir = None):
        """See base class."""
        return self.get_examples_from_file(
            os.path.join(data_dir, "test.jsonl"), "test")

    def get_labels(self):
        """See base class."""
        raise NotImplementedError

    def _create_examples(self, lines, set_type, bias_lines=None, teach_lines = None):
        raise NotImplementedError
