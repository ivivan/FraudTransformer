#!/usr/bin/env python
# coding: utf-8

# importing python utility libraries
import os, sys, random, io, urllib
from datetime import datetime


os.environ['http_proxy'] = "http://squid-a10.prod.ice.int.threatmetrix.com:3128"
os.environ['https_proxy'] = "http://squid-a10.prod.ice.int.threatmetrix.com:3128"

from pathlib import Path
from utils.data_config import input_config
from utils.fraud_data import FraudData
from utils.tunning_related import prepare_pytorch_data,print_summary,compute_metrics

# importing pytorch libraries
import torch
from torch import nn
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# peft package, for lora training
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PeftModel,
    PeftConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

# huggingface related
from datasets import Dataset
from transformers import MobileBertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, logging
from transformers import pipeline, TextClassificationPipeline
from transformers import BertTokenizer, BertModel
from transformers import AdamW,get_linear_schedule_with_warmup
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


from sklearn.model_selection import train_test_split

# Tracking
from aim.hugging_face import AimCallback


# importing data science libraries
import pandas as pd
import numpy as np
import pickle as pkl


# importing python plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# ### 1.2 CPU/GPU Device
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print(device)

### 1.4 Random Seed Initialization

# Finally, let' set the seeds of random elements in the code e.g. the initialization of the network parameters to guarantee deterministic computation and results:
# init deterministic seed
seed_value = 6758
random.seed(seed_value)  # set random seed
np.random.seed(seed_value)  # set numpy seed
torch.manual_seed(seed_value)  # set pytorch seed CPU
if (torch.backends.cudnn.version() != None):
    torch.cuda.manual_seed(seed_value)  # set pytorch seed GPU

## 2. Fraud Data
# use new fraud data prepared from trustscore customers 2023


### 2.1 Load the Parsed Data

#### load final features ######
with open('/home/zhanyi02/FeatureEngineering/checkpoints/featureimportance/plots/new/final_selected_features.npy',
          'rb') as f:
    vars_rc_list = np.load(f)

vars_rc_list = vars_rc_list.tolist()

vars_rc_list =  [x for x in vars_rc_list if  x.startswith(('tsrc','tmxrc'))]

print('total var length:', len(vars_rc_list))

csrt_train = pd.read_csv('/data/zhanyi02/trustscore/trustscore2023/combined/trust_score_2023_train.csv',usecols=vars_rc_list+['frd'])
print('train_dataset_size', csrt_train.shape)

# # original : 8% from each group, use the subset to fine-tune the classification model
# train_data_subset = csrt_train.groupby('frd').apply(lambda x: x.sample(frac=0.05)).reset_index(drop=True)

train_df, val_df = train_test_split(csrt_train, test_size=0.3, random_state=seed_value)



### 2.2 Check Data
X_train = train_df[vars_rc_list]
X_valid = val_df[vars_rc_list]
y_train = train_df['frd']
y_valid = val_df['frd']

vocab_file = "./checkpoints/nlp2023/v1/vocab.txt"
## do not rerun this code chunk

vocab = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]', '[BOS]', '[EOS]']
for feature in vars_rc_list:
    vocab.append(feature + "_0")
    vocab.append(feature + "_1")

with open(vocab_file, 'w') as vocab_fp:
    vocab_fp.write("\n".join(vocab))

#### Tokenizer

special_tokens_dict = {"unk_token": "[UNK]",
                       "sep_token": "[SEP]",
                       "pad_token": "[PAD]",
                       "cls_token": "[CLS]",
                       "mask_token": "[MASK]",
                       "eos_token": "[EOS]",
                       "bos_token": "[BOS]"}

tokenizer = MobileBertTokenizer(vocab_file, do_basic_tokenize=False)
tokenizer.add_special_tokens(special_tokens_dict)



#### Transform Data
train_data_features = X_train.copy()
train_data_labels = y_train

##### fro validation
val_data_features = X_valid.copy()
val_data_labels = y_valid


#### Preprocess Data

vocab_dict = tokenizer.get_vocab()

##### prepare train test
train_df = prepare_pytorch_data(train_data_features, train_data_labels, vocab_dict, device)
val_df = prepare_pytorch_data(val_data_features, val_data_labels, vocab_dict, device)

# #### Fine-Tuning Pretrained Model
#
# clsmodel = BertForSequenceClassification.from_pretrained('./checkpoints/nlp2023/v2/checkpoint-36000', num_labels=2).to(device)
#
# num_epochs = 20
# lr = 2e-5
#
# arguments = TrainingArguments(
#     output_dir="./checkpoints/nlp2023/v2/finetuning", # output directory
#     per_device_train_batch_size=32, # batch size per device during training
#     per_device_eval_batch_size=32,  # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     num_train_epochs=num_epochs,     # total # of training epochs
#     evaluation_strategy="epoch",     # run validation at the end of each epoch
#     save_strategy="epoch",
#     save_total_limit=2,
#     learning_rate=lr,
#     load_best_model_at_end=True,
#     seed=seed_value,
#     report_to="wandb"
# )
#
#
# trainer = Trainer(
#     model=clsmodel,
#     args=arguments,
#     train_dataset=train_df,
#     eval_dataset=val_df, # change to test when you do your final evaluation!
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
#
# result = trainer.train()
#
# print_summary(result)



#### lora fine-tuning
batch_size = 32
model_name_or_path = './checkpoints/nlp2023/v2/checkpoint-36000'
task = "mrpc"
peft_type = PeftType.LORA
num_epochs = 20
lr = 2e-5

peft_config = LoraConfig(task_type="SEQ_CLS",target_modules=["query", "value"], inference_mode=False, r=4, lora_alpha=8, lora_dropout=0.1)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2, return_dict=True).to(device)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

arguments = TrainingArguments(
    output_dir="./checkpoints/nlp2023/v2/lora", # output directory
    per_device_train_batch_size=batch_size, # batch size per device during training
    per_device_eval_batch_size=batch_size,  # batch size for evaluation
    warmup_steps=2000,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    num_train_epochs=num_epochs,     # total # of training epochs
    evaluation_strategy="epoch",     # run validation at the end of each epoch
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=lr,
    load_best_model_at_end=True,
    seed=seed_value,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_df,
    eval_dataset=val_df, # change to test when you do your final evaluation!
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

result = trainer.train()

# Save our LoRA model & tokenizer results
peft_model_id="./checkpoints/nlp2023/v2/lora/best_lora"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

print_summary(result)