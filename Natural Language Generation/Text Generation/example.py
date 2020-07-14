# coding: utf-8
import re
import torch
import argparse
import getpass
import numpy as np
from transformers import *
from torch.nn import Softmax
from time import gmtime, strftime
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, file_path):

        with open(file_path, 'r') as f:
            lines = f.read().splitlines()

        samples = []
        for line in lines:

            try:
                input_sentence = re.search('("(.)+")', line).group(1)

            except AttributeError:
                print("A sample may not be well-formatted:\n{}".format(line))
                continue

            target_sentence = line.replace(input_sentence, '')
            samples.append([input_sentence, target_sentence[1:]])  # removing the ',' at index 0

        self.lines = np.asarray(samples)
        print(self.lines.shape)
        self.sources_preprocessed = tokenizer(self.lines[:, 0].tolist(), padding=True, truncation=True,
                                              return_tensors="pt")
        self.targets_preprocessed = tokenizer(self.lines[:, 1].tolist(), padding=True, truncation=True,
                                              return_tensors="pt")

        self.labels = self.targets_preprocessed['input_ids'][:, 1:]
        self.targets_preprocessed['input_ids'] = self.targets_preprocessed['input_ids'][:, :-1]
        self.targets_preprocessed['attention_mask'] = self.targets_preprocessed['attention_mask'][:, :-1]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return {'input_ids': self.sources_preprocessed['input_ids'][idx],
                'attention_mask': self.sources_preprocessed['attention_mask'][idx],
                'decoder_input_ids': self.targets_preprocessed['input_ids'][idx],
                'decoder_attention_mask': self.targets_preprocessed['attention_mask'][idx],
                'labels': self.labels[idx]
                }


if __name__ == '__main__':

    # TODO argparse

    experiment = getpass.getuser() + ' ' + strftime("%Y-%m-%d %H:%M:%S", gmtime())
    output_dir = './results/' + experiment + '/'
    training_file = './data/train/trainset.csv'
    evaluation_file = './data/val/devset.csv'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Config
    config_encoder = BertConfig
    config_decoder = BertConfig()

    encoder = BertModel.from_pretrained('bert-base-uncased')
    decoder = BertForMaskedLM.from_pretrained('bert-base-uncased')

    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    training_args_dict = {"output_dir": output_dir,
                          "overwrite_output_dir": False,
                          "do_train": True,
                          "do_eval": True,
                          "do_predict": False,
                          "evaluate_during_training": False,
                          "per_device_train_batch_size": int(1),
                          "per_device_eval_batch_size": int(1),
                          "gradient_accumulation_steps": int(1),
                          "learning_rate": float(5e-5),
                          "weight_decay": float(0),
                          "adam_epsilon": float(1e-8),
                          "max_grad_norm": float(1.0),
                          "num_train_epochs": float(3.0),
                          "max_steps": int(-1),
                          "warmup_steps": int(0),
                          # "logging_dir": 'run/runs/**CURRENT_DATETIME_HOSTNAME**',
                          "logging_first_step": False,
                          "logging_steps": int(500),
                          "save_steps": int(500),
                          "save_total_limit": int(100),
                          "no_cuda": True,
                          "seed": int(42),
                          "fp16": False,
                          "fp16_opt_level": 'O1',
                          "local_rank": int(-1),
                          "tpu_num_cores": int(0),
                          # "debug": False,
                          # "dataloader_drop_last": False,
                          # "eval_steps": int(1000),
                          # "past_index": -1
                          }

    training_args = TrainingArguments(**training_args_dict)
    training_dataset = MyDataset(training_file)
    val_dataset = MyDataset(evaluation_file)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        # data_collator=data_collator,
        train_dataset=training_dataset,
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    # Evaluation
    if training_args.do_eval:
        preds = trainer.predict(val_dataset)
        # preds.predictions are logits !

        # TODO
        normalizer = Softmax(dim=-1)
        preds_probs = normalizer(torch.FloatTensor(preds.predictions))
        preds_ids = torch.argmax(preds_probs, dim=-1)
        for idx in range(preds_ids.size()[0]):
            print(tokenizer.convert_ids_to_tokens(preds_ids[idx].tolist()))
