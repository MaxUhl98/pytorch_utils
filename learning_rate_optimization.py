"""Chris Deotts Learning rate optimization technique automatized
Source: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/488083#2727164
Actually there is a small difference in my implementation to Chris description: This program does not train one more epoch then the best val loss, it trains until it gets the best val loss and then reduces the lr by 10
Also I want to say a HUGE thank you to Chris Deotte for all his contributions to HMS brain Activity Classification, which where all very informative and helped me grow as a Data Scientist"""
import os
import uuid

import torch
from torch import nn
from typing import Optional, Union, Dict, Any, Callable, Iterable, Tuple, List
from engine import train, train_step, test_step
from torch.utils.data import DataLoader
from helpers import get_logger
import numpy as np
from _config import Config
import pandas as pd
from SelfNet import SelfNet
from data_generator import GenericData, prepare_train


def reset_model(model: nn.Module, reset_weight_path: str) -> None:
    """Resets the weights of a model to the weights at reset_weight_path

    :param model: Model to be reset
    :param reset_weight_path: Weights used to reset model
    :return: None (resets model inplace)
    """
    model.load_state_dict(torch.load(reset_weight_path))


def find_best_start_lr(model: nn.Module, optim: Callable, train_loader: DataLoader, test_loader: DataLoader,
                       loss_fn: nn.Module, max_epochs: int = 15, tested_lr_list=None, patience: int = 3,
                       device: Union[str, torch.device] = 'cuda') -> Tuple[float, List[float]]:
    """

    :param model:
    :param optim:
    :param train_loader:
    :param test_loader:
    :param loss_fn:
    :param max_epochs:
    :param tested_lr_list:
    :param patience:
    :param device:
    :return:
    """
    if tested_lr_list is None:
        tested_lr_list = [1e-6,5e-6,1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    run_id = uuid.uuid1()
    reset_weight_path = f'{run_id}.pth'
    torch.save(model.state_dict(), reset_weight_path)
    logger = get_logger(f'LR_TEST_{run_id}')
    lr_data = {}
    best_val = (10 ** 10, -1)
    for lr in tested_lr_list:
        optimizer = optim(model.parameters(), lr=lr)
        results = train(model, train_loader, test_loader, optimizer, loss_fn, max_epochs, patience=patience,
                        save_best=False,
                        logger=logger, device=device)
        val = min(results['val_loss'])
        best_val = (val, lr) if val < best_val[0] else best_val
        lr_data[lr] = lr, results['val_loss']
        reset_model(model, reset_weight_path)
    return lr_data[best_val[1]]


def find_lr_steps(lr: float, model: nn.Module, optim: Callable,
                  train_loader: DataLoader, test_loader: DataLoader,
                  loss_fn: nn.Module, patience: int = 3,
                  device: Union[str, torch.device] = 'cuda', divisor: int = 10):
    """

    :param lr:
    :param model:
    :param optim:
    :param train_loader:
    :param test_loader:
    :param loss_fn:
    :param patience:
    :param device:
    :param divisor:
    :return:
    """
    run_id = uuid.uuid1()
    reset_weight_path = f'D:/models/HMS/{run_id}.pth'
    torch.save(model.state_dict(), reset_weight_path)
    logger = get_logger(f'LR_TEST_{run_id}')
    optimizer = optim(model.parameters(), lr=lr)
    best_val = 10 ** 10
    global_best_val = 10 ** 10
    cnt = 0
    is_improving = True
    total_count = 0
    lr_step_dict = {}

    while is_improving:
        is_improving = False
        while 1:
            train_step(model, train_loader, loss_fn, optimizer, device, False, logger)
            val = test_step(model, test_loader, loss_fn, device)
            if val < best_val:
                best_val = val
                cnt = 0
                if val < global_best_val:
                    global_best_val = val
                    torch.save(model.state_dict(), reset_weight_path)
                    is_improving = True
            if cnt == patience:
                break
            cnt += 1
            total_count += 1
        reset_model(model, reset_weight_path)
        lr_step_dict[lr] = total_count
        lr /= divisor
    return lr_step_dict, global_best_val


if __name__ == '__main__':
    os.chdir('..')
    name = 'SelfNet_lr_test'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    mlp_activation = nn.RReLU
    conv_activation = nn.RReLU
    expected_shape = Config.full_data_expected_shape
    model_kwargs = {
        'kernel_size': 3,
        'spec_inner_dim': 16,
        'spec_shape': expected_shape,
        'n_heads': 8,
        'mlp_dim': 32,
        'pooler_kernel_size': 2,
        'pooler_kwargs': {'stride': 2},
        'linear_input_size': 625,
        'attn_dropout': .5,
        'dropout': .3,
        'mlp_activation_fn': mlp_activation,
        'conv_activation': conv_activation
    }
    model = SelfNet(**model_kwargs)
    logger = get_logger(name)
    df = pd.read_csv(Config.train_path)
    df = prepare_train(df)

    # Nearly 80-20 split using patient ids
    df_train, df_test = df.loc[df.patient_id>14000], df.loc[df.patient_id<=14000]

    specs = np.load(Config.full_data_path, allow_pickle=True).item()

    train_data, test_data = GenericData(df=df_train, data_dict=specs, expected_shape=expected_shape), GenericData(
        df=df_test, data_dict=specs, expected_shape=expected_shape)
    train_loader, test_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=False), DataLoader(
        test_data,
        batch_size=Config.batch_size, shuffle=False)
    # eegs = np.load(expert_eeg_path, allow_pickle=True).item()

    # data = GenericData(df=df, data_dict=specs, expected_shape=spectrogram_expected_shape)

    # Create Optimizer and LR-Scheduler
    optim = torch.optim.AdamW
    # optim = CosineAnnealingWarmRestarts(optimizer=optim, **Config.cosanneal_res_params)

    # Define Loss Function
    loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    lr, results = find_best_start_lr(model=model, optim=optim, train_loader=train_loader, test_loader=test_loader,loss_fn=loss_fn)
    logger.info(f'best starting LR found is {lr}/nAll results:{results}')
    steps, best_loss = find_lr_steps(lr, model=model,optim=optim,train_loader=train_loader,test_loader=test_loader,loss_fn=loss_fn)
    logger.info(f'Reached best val_loss:{best_loss} with steps: {steps}')


