import torch
import numpy as np
from time import time
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, \
    f1_score, confusion_matrix


def train(model, criterion, optimizer, train_loader, device):
    print("Training...")
    model.train()
    losses = list()
    pred = list()
    gt = list()

    for data, label in tqdm(train_loader):
        data = data.float().to(device)
        label = label.long().to(device)
        output = model(data)
        loss = criterion(output, label)
        output_label = torch.max(output, 1).indices.cpu().detach().numpy()
        gt_label = label.cpu().detach().numpy()
        losses.append(loss.item())
        for i in range(output_label.shape[0]):
            pred.append(round(output_label[i]))
            gt.append(round(gt_label[i]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_loss = np.mean(losses)
    print("Loss:", mean_loss)
    acc = accuracy_score(gt, pred)
    print("Acc:", acc)
    f1 = f1_score(gt, pred, average="macro")
    print("Macro avg F1 score:", f1)
    print(classification_report(gt, pred))
    conf_matrix = confusion_matrix(gt, pred)

    return f1, acc, mean_loss, conf_matrix

def evaluate(model, criterion, val_loader, device, log=True):
    print("Evaluating...")
    model.eval()
    losses = list()
    pred = list()
    gt = list()

    for data, label in tqdm(val_loader):
        data = data.float().to(device)
        label = label.long().to(device)
        output = model(data)
        loss = criterion(output, label)
        output_label = torch.max(output, 1).indices.cpu().detach().numpy()
        gt_label = label.cpu().detach().numpy()
        losses.append(loss.item())
        for i in range(output_label.shape[0]):
            pred.append(round(output_label[i]))
            gt.append(round(gt_label[i]))

    mean_loss = np.mean(losses)
    if log:
        print("Loss:", mean_loss)
    acc = accuracy_score(gt, pred)
    if log:
        print("Acc:", acc)
    f1 = f1_score(gt, pred, average="macro")
    if log:
        print("Macro avg F1 score:", f1)
    clf_report = classification_report(gt, pred)
    if log:
        print(clf_report)
    conf_matrix = confusion_matrix(gt, pred)

    return f1, acc, clf_report, mean_loss, conf_matrix


def infer(model, data, device, speed_test_times=1):
    print("Predicting...")
    model.eval()

    data = data.float().to(device)
    speeds = list()
    for _ in range(speed_test_times):
        begin = time()
        output = model(data)
        speeds.append(time()-begin)
    output_label = torch.max(output, 1).indices.cpu().detach().numpy()
    output_prob = torch.softmax(output, 1).cpu().detach().numpy()

    return output_label, output_prob, output, np.mean(speeds)
    