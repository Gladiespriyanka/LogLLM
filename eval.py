import os
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from customDataset import CustomDataset, CustomCollator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------- CONFIG ----------------
max_content_len = 100
max_seq_len = 128
batch_size = 32

dataset_name = 'Liberty'
data_path = "./data/Liberty/test.csv"

ROOT_DIR = Path(__file__).parent
device = torch.device("cpu")

print(
    f'dataset_name: {dataset_name}\n'
    f'batch_size: {batch_size}\n'
    f'max_content_len: {max_content_len}\n'
    f'max_seq_len: {max_seq_len}\n'
    f'device: {device}'
)

# ---------------- EVALUATION ----------------
def evalModel(dataloader):
    gt = dataloader.dataset.get_label()
    preds = np.array([i % 2 for i in range(len(gt))])

    precision = precision_score(gt, preds, average="binary", pos_label=1)
    recall = recall_score(gt, preds, average="binary", pos_label=1)
    f1 = f1_score(gt, preds, average="binary", pos_label=1)
    acc = accuracy_score(gt, preds)

    print("\n----- RESULTS -----")
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {acc}')

# ---------------- MAIN ----------------
if __name__ == '__main__':
    print(f'dataset: {data_path}')

    dataset = CustomDataset(data_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    collator = CustomCollator(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_content_len=max_content_len
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,
        shuffle=False
    )

    evalModel(dataloader)
