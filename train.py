import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from Bert import BertConfig, Bert
from transformers import BertTokenizer
import jieba
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
lr = 1e-5
epochs = 30
config = BertConfig()
train_size = 0.8
tokenizer = BertTokenizer.from_pretrained("bert_tokenizer")
config.num_labels = 5
config.vocab_size = 30521
label_ls = ["False", "True"]

def read_data():

    paths = glob(os.path.join('/scratch/s5497094/sarcasm/video_tensor', '*'))

    train_paths, test_paths = train_test_split(paths, train_size=train_size)

    return train_paths, test_paths

class CustomData(Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __getitem__(self, item):
        video_path = self.datas[item]

        name = os.path.splitext(os.path.basename(video_path))[0]

        audio_path = os.path.join('/scratch/s5497094/sarcasm/audio_tensor', f"{name}.pkl")
        text_path = os.path.join('/scratch/s5497094/sarcasm/texts', f"{name}.txt")

        video_value = torch.load(video_path)
        audio_value = torch.load(audio_path)

        with open(text_path, encoding="utf-8") as f:
            texts = f.read()

        text, label = texts.split("\t")
        label = label_ls.index(label)

        return text, label, video_value, audio_value

    @staticmethod
    def call_fc(batch):

        texts = []
        labels = []
        videos = []
        wavs = []

        for x, y, video_value, wav_value in batch:
            texts.append(x)
            labels.append(y)
            videos.append(video_value)
            wavs.append(wav_value)

        inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation="only_first", max_length=1024, return_tensors="pt")
        labels = np.array(labels, dtype="int64")

        labels = torch.from_numpy(labels)

        inputs["video_values"] = torch.cat(videos, dim=0)
        inputs["audio_values"] = torch.cat(wavs, dim=0)

        return inputs, labels

    def __len__(self):
        return len(self.datas)

def train():
    train_data, test_val_data = read_data()

    test_data, val_data = train_test_split(test_val_data, test_size=0.5)

    train_data = CustomData(train_data)
    train_data = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=train_data.call_fc)

    val_data = CustomData(val_data)
    val_data = DataLoader(val_data, shuffle=True, batch_size=batch_size, collate_fn=val_data.call_fc)
    
    test_data = CustomData(test_data)
    test_data = DataLoader(test_data, shuffle=True, batch_size=batch_size, collate_fn=test_data.call_fc)

    model = Bert(config)
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fc = nn.CrossEntropyLoss()

    loss_old = 100

    train_result = []
    val_result = []
    test_result = []  # Initialize test_result list here

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_data)
        loss_all = 0
        acc_all = 0
        for step, (x, y) in enumerate(pbar):
            x = {k:v.to(device) for k, v in x.items()}
            y = y.to(device)
            out = model(**x)

            loss = loss_fc(out, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_all += loss.item()
            loss_time = loss_all / (step + 1)

            acc = torch.mean((y == torch.argmax(out, dim=-1)).float())

            acc_all += acc
            acc_time = acc_all / (step + 1)

            s = "train => epoch:{} - step:{} - loss:{:.4f} - loss_time:{:.4f} - acc:{:.4f} - acc_time:{:.4f}".format(epoch, step, loss, loss_time, acc, acc_time)
            pbar.set_description(s)

            train_result.append(s+"\n")

        # Validation loop
        val_loss_all = 0
        val_acc_all = 0
        val_steps = 0
        for x, y in val_data:
            x = {k:v.to(device) for k, v in x.items()}
            y = y.to(device)
            out = model(**x)

            loss = loss_fc(out, y)

            val_loss_all += loss.item()
            val_acc_all += torch.mean((y == torch.argmax(out, dim=-1)).float())
            val_steps += 1

        val_loss = val_loss_all / val_steps
        val_acc = val_acc_all / val_steps

        val_result.append(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

        with torch.no_grad():
            pbar = tqdm(test_data)
            loss_all = 0
            acc_all = 0
            for step, (x, y) in enumerate(pbar):
                x = {k: v.to(device) for k, v in x.items()}
                y = y.to(device)
                out = model(**x)

                loss = loss_fc(out, y)

                loss_all += loss.item()
                test_loss_time = loss_all / (step + 1)

                acc = torch.mean((y == torch.argmax(out, dim=-1)).float())

                acc_all += acc
                acc_time = acc_all / (step + 1)

                s = "test => epoch:{} - step:{} - loss:{:.4f} - loss_time:{:.4f} - acc:{:.4f} - acc_time:{:.4f}".format(epoch, step, loss, test_loss_time, acc, acc_time)
                pbar.set_description(s)

                test_result.append(s+"\n")  # Append test results to test_result list

        with open("train_result.txt", "w") as f:
            f.writelines(train_result)

        with open("val_result.txt", "w") as f:
            f.writelines(val_result)

        with open("test_result.txt", "w") as f:
            f.writelines(test_result)  # Write test_result to file

        if loss_old > test_loss_time:
            loss_old = test_loss_time
            torch.save(model.state_dict(), "model.pkl")

    ys = []
    prs = []

    model.load_state_dict(torch.load("model.pkl"))
    model.eval()

    with torch.no_grad():
        pbar = tqdm(test_data)
        for step, (x, y) in enumerate(pbar):
            x = {k:v.to(device) for k, v in x.items()}
            y = y.to(device)
            out = model(**x)

            p = torch.argmax(out, dim=-1).cpu().numpy()
            y = y.cpu().numpy()

            ys.append(y)
            prs.append(p)

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(prs, axis=0)

    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"precision：{precision*100}%")
    print(f"recall：{recall*100}%")
    print(f"f1：{f1*100}%")


    cm = confusion_matrix(y_true, y_pred)
    true_labels = np.unique(y_true)

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(len(true_labels)), true_labels)
    plt.yticks(np.arange(len(true_labels)), true_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion matrix')

    for i in range(len(true_labels)):
        for j in range(len(true_labels)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

    plt.show()

if __name__ == '__main__':
    train()

