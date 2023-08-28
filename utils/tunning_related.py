import os
import pandas as pd
import numpy as np
from pynvml import *
import torch

from datasets import Dataset

from sklearn.metrics import roc_auc_score,precision_recall_fscore_support,accuracy_score

def prepare_pytorch_data(features, labels, vocab_dict, device):
    # use this one for final implementation
    for column in features:
        features[column] = features[column].map(lambda x: column + "_" + str(x))
        features[column] = features[column].map(lambda x: vocab_dict.get(x, vocab_dict['[UNK]']))

    input_initial = features.to_numpy().tolist()

    inputs = [[vocab_dict['[CLS]']] + x + [vocab_dict['[SEP]']] for x in input_initial]

    my_dict = {'input_ids': inputs,
               'labels': labels.to_list(),
               'token_type_ids': [[0] * (len(features.columns) + 2)] * len(features),
               'attention_mask': [[1] * (len(features.columns) + 2)] * len(features)}

    prepared_df = Dataset.from_dict(my_dict)

    tf_df = prepared_df.with_format("torch", device=device)

    return tf_df


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def generate_embeddings(data_loader, model):
    # Evaluate the dataset
    temp = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids']  # Add batch dimension
            attention_mask = batch['attention_mask'] # Add batch dimension

            # Move the input tensors to the same device as the model
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

            model_output = model(input_ids, attention_mask=attention_mask)

            sentence_embedding = torch.mean(model_output.hidden_states[-1], dim=1).squeeze()
            temp.append(sentence_embedding)
    
    results = torch.cat(temp, axis=0)
    
    return results.tolist()


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    roc_auc = roc_auc_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': roc_auc
    }