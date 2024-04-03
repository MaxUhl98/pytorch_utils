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

def get_logger(name:str, base_filepath:str='logs/model_experiments') -> logging.Logger:
    """Creates a logging.Logger object that writes a logfile named name.log into the folder at base_filepath
    (throws an error if the folder does not exist)

    :param name: Name of the logger and the logging file
    :param base_filepath: Path to the folder in which the logging file will get saved
    :return: Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'{base_filepath}/{name}.log', encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


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
    """Tests all initial learning rates out of tested_lr_list
    (default=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    Returns the learning rate that had the lowest validation loss when training with the data inside the train_loader
    and validating using the data in val_loader. Trains for max_epoch steps, cancels the training if the validation
    loss did not increase in the last patience epochs.


    :param model: Model for which the best initial learning rate shall be found
    :param optim: Optimizer to use for training
    :param train_loader: DataLoader containing training data
    :param test_loader: DataLoader containing the validation data
    :param loss_fn: Function used to calculate the loss
    :param max_epochs: Maximum number of epochs to train for one learning rate trial
    :param tested_lr_list: List containing the numeric values of the learning rates that shall be tested
    :param patience: A trial will get canceled if the validation loss did not improve in the last patience steps
    :param device: Pytorch Device to train the model on
    :return: Best performing learning rate from the trials and
     list containing the validation loss of each step from the trial with the best learning rate
    """
    if tested_lr_list is None:
        tested_lr_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    run_id = uuid.uuid1()
    reset_weight_path = f'{run_id}.pth'
    torch.save(model.state_dict(), reset_weight_path)
    logger = get_logger(f'LR_TEST_{run_id}')
    logger.log(f'Starting lr optimization run with ID {run_id}')
    lr_data = {}
    best_val = (10 ** 10, -1)
    for lr in tested_lr_list:
        logger.info(f'Testing learning rate {lr}')
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
                  loss_fn: nn.Module, patience: int = 5,global_patience:int=20,
                  device: Union[str, torch.device] = 'cuda', divisor: int = 10):
    """Uses a given learning rate to determine the steps at which the learning rate should be decayed.
    Uses trains the model using hte optimizer declared at optim with the data inside train_loader as training data and
     uses the data inside val_loader as validation data. Calculates the losses with the loss_fn.
     Trains the model on the device declared at device. Divides the learning rate by divisor at each step.

    :param lr: Initial learning rate used
    :param model: Model for which the learning rate schedule shall be determined
    :param optim: Optimizer used for model training
    :param train_loader: DataLoader containing the training data
    :param test_loader: DataLoader containing the validation data
    :param loss_fn: FUnctions used to calculate the losses
    :param patience: A trial will get canceled if the validation loss did not improve in the last patience steps
    :param device: Pytorch Device to train the model on
    :param divisor: Number the learning rate will get divided by if it did not improve the val loss in the last patience steps
    :return: Dictionary containing the learning rates and the number of last epoch in the trial where they reduced the val loss
    """
    assert patience >= 1, AssertionError(
        f'We need a patience of at least 1 for this method, because the model 1 training epoch after the best will get taken for the next lr decay trial')
    run_id = uuid.uuid1()
    reset_weight_path = f'D:/models/HMS/{run_id}.pth'
    torch.save(model.state_dict(), reset_weight_path)
    logger = get_logger(f'LR_TEST_{run_id}')
    logger.info(f'Starting lr-step trial with ID: {run_id}')
    optimizer = optim(model.parameters(), lr=lr)
    best_val = 10 ** 10
    global_best_val = 10 ** 10
    cnt = 0
    total_count = 0
    global_patience_count = 0
    lr_step_dict = {}
    save_next = False
    go_big = True

    while go_big:
        go = True
        while go:
            train_val = train_step(model, train_loader, loss_fn, optimizer, device, False, logger)
            val = test_step(model, test_loader, loss_fn, device)
            logger.info(
                f"Learning Rate: {lr} | "
                f"train_loss: {train_val:.4f} | "
                f"val_loss: {val:.4f} | "
            )
            if save_next:
                torch.save(model.state_dict(), reset_weight_path)
                save_next = False
            if val < best_val:
                best_val = val
                cnt = 0
                if val < global_best_val:
                    global_best_val = val
                    torch.save(model.state_dict(), reset_weight_path)
                    global_patience_count = 0
                    save_next = True
            if cnt == patience:
                go = False
            if global_patience_count == global_patience:
                return lr_step_dict, global_best_val
            cnt += 1
            total_count += 1
            global_patience_count += 1
        reset_model(model, reset_weight_path)
        lr_step_dict[lr] = total_count
        lr /= divisor


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
    df_train, df_test = df.loc[df.patient_id > 14000], df.loc[df.patient_id <= 14000]

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
    # lr, results = find_best_start_lr(model=model, optim=optim, train_loader=train_loader, test_loader=test_loader,
    #                                 loss_fn=loss_fn)
    # logger.info(f'best starting LR found is {lr}/nAll results:{results}')
    steps, best_loss = find_lr_steps(0.001, model=model, optim=optim, train_loader=train_loader,
                                     test_loader=test_loader,
                                     loss_fn=loss_fn)
    logger.info(f'Reached best val_loss:{best_loss} with steps: {steps}')
