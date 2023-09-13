import torch
from torch import nn
from helpers import accuracy_fn
import matplotlib.pyplot as plt
import time
"""
Most Credit to https://github.com/mrdbourke/pytorch-deep-learning
I implemented these Functions while taking the ZTM Pytorch Deep Learning Course
"""
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
def train_step(model:torch.nn.Module,data_loader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,optimizer:torch.optim.Optimizer,device : torch.device):
    """Vanilla Training Epoch
    Returns Train loss, Train Accuracy"""
    acc = 0
    train_loss = 0
    for batch, (X,y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        model.train()
        pred = model(X)
        acc += accuracy_fn(y, pred.argmax(dim=1))
        loss = loss_fn(pred,y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print()
    print(f'Train Loss: {loss/len(data_loader):.5f} Train Acc: {acc/len(data_loader):.2f}')
    return loss/len(data_loader),acc/len(data_loader)

def test_step(model:torch.nn.Module,data_loader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,device : torch.device):
    """Vanilla Test Epoch
     Returns Val Loss, Val Acc, Model params"""
    test_loss,test_acc = 0,0
    model.eval()
    with torch.no_grad():
        for X,y in data_loader:
            X,y = X.to(device),y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred,y)
            test_acc += accuracy_fn(y,test_pred.argmax(dim=1))

        print()
        print(f"Val Loss: {test_loss/len(data_loader):.5f} Val Acc: {test_acc/len(data_loader):.2f}")
        #Short waiting time required to keep console output consistent
        time.sleep(.1)
    return test_loss/len(data_loader), test_acc/len(data_loader), model.parameters()

