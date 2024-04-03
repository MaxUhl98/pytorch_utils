"""
Contains functions for training and testing a PyTorch model.
Based on https://github.com/mrdbourke/pytorch-deep-learning, 
got refined by me over time by adding average meters, gradient clipping, logging, including k-folds, etc.
Is currently made for kl div loss, but can easily be changed for different applications
"""
import logging
import sys
import uuid

import numpy as np

from data_generator import GenericData
import pandas as pd
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Union, Any, Iterable, Callable
from torch.nn import Sigmoid, Softmax
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Transform
import random
from _config import Config
from torch.optim.lr_scheduler import OneCycleLR


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device, lr_scheduling: bool, logger: logging.Logger, omni_vec: bool = False,
               vit: bool = False,
               need_softmax: bool = False, use_average_meter: bool = True, patchtst: bool = False,
               patchtsmixer: bool = False, model_input_type: str = 'singular') -> float:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=Config.apex)
    avg_meter = AverageMeter() if use_average_meter else None

    # Setup train loss and train accuracy values
    train_loss = 0
    # Loop through data loader data batches
    logger.info('Starting training...')
    for batch in dataloader:
        # Send data to target device

        if model_input_type == 'singular':
            X, y = batch
            X = [X.to(device)]
            y = y.to(device)
        else:
            X, y = batch[:-1], batch[-1]
            X = [x.to(device) for x in X]
        if vit:
            with torch.cuda.amp.autocast(enabled=Config.apex):
                y_pred = model(*X)
            y_pred = F.log_softmax(y_pred.logits, dim=1)
        elif need_softmax:
            with torch.cuda.amp.autocast(enabled=Config.apex):
                y_pred = F.log_softmax(model(*X), dim=1)
        elif patchtst:
            with torch.cuda.amp.autocast(enabled=Config.apex):
                y_pred = model(*X)
            y_pred = F.log_softmax(y_pred.prediction_logits, dim=1)
        elif patchtsmixer:
            with torch.cuda.amp.autocast(enabled=Config.apex):
                y_pred = model(*X)
            y_pred = F.log_softmax(y_pred.prediction_outputs, dim=1)
        else:
            with torch.cuda.amp.autocast(enabled=Config.apex):
                y_pred = torch.log(model(*X))
        if y_pred.isnan().sum() != 0:
            logger.info('Nan values encountered, stopping training')
            break
            # logger.info(y_pred)
            # logger.info(model(*X))
            # logger.info(optimizer)
            # logger.info(type(X))
            # logger.info(X.isnan().sum())
            # logger.info(X.shape)
            # y = model(X.to(device))
            # logger.info(y)
            # logger.info(torch.log(y))
            # with torch.cuda.amp.autocast(enabled=Config.apex):
            #    y_auto = torch.log(model(X.to(device)))
            # logger.info(y_auto)
            # logger.info(f'Omnicevec:{omni_vec}')
            # logger.info(f'ViT:{vit}')
            # logger.info(f'Need Softmax:{need_softmax}')
            # logger.info(f'PatchTST:{patchtst}')
            # logger.info(f'PatchTSMixer:{patchtsmixer}')
            # for name, param in model.named_parameters():
            #    logger.info(name, torch.isfinite(param.grad).all())
            # sys.exit()

        # 2. Calculate  and accumulate loss
        with torch.cuda.amp.autocast(enabled=Config.apex):
            loss = loss_fn(y_pred, y.float())
            train_loss += loss.item()

        # 3. Optimizer zero grad
        if lr_scheduling:
            optimizer.optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        # 4. Loss backward
        scaler.scale(loss).backward()

        # 4.5 gradient accumulation & clipping
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Averaging
        if avg_meter:
            avg_meter.update(loss.item(), y_pred.shape[0])

        # 5. Optimizer step
        if lr_scheduling:
            scaler.step(optimizer.optimizer)
            optimizer.step()
            scaler.update()
        else:
            scaler.step(optimizer)
            scaler.update()
        # logger.info(f'Loss in batch {batch}: {loss}')

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    return avg_meter.avg if avg_meter else train_loss


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device,
              vit: bool = False,
              need_softmax: bool = False, patchtst: bool = False, patchtsmixer: bool = False,
              model_input_type: str = 'singular', use_average_meter: bool = True) -> float:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (val_loss, val_accuracy). For example:

    (0.0223, 0.8985)
    """
    model.eval()
    # Setup test loss and test accuracy values
    val_loss = 0

    loss_avg_meter = AverageMeter() if use_average_meter else None
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch in dataloader:
            # Send data to target device
            if model_input_type == 'singular':
                X, y = batch
                X = [X.to(device)]
                y = y.to(device)
            else:
                X, y = batch[:-1], batch[-1]
                X = [x.to(device) for x in X]
            if vit:
                val_pred_logits = F.log_softmax(model(*X).logits, dim=1)
            elif need_softmax:
                val_pred_logits = F.log_softmax(model(*X), dim=1)
            elif patchtst:
                val_pred_logits = F.log_softmax(model(*X).prediction_logits, dim=1)
            elif patchtsmixer:
                val_pred_logits = F.log_softmax(model(*X).prediction_outputs, dim=1)
            else:
                val_pred_logits = torch.log(model(*X))
            # 2. Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y.float())
            val_loss += loss.item()
            if loss_avg_meter:
                loss_avg_meter.update(loss.item(), val_pred_logits.shape[0])

    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    return loss_avg_meter.avg if use_average_meter else val_loss


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: Union[torch.device, str], logger: logging.Logger, lr_scheduling: bool = False,
          patience: int = 10000,
          save_best: bool = True, save_path: str = 'model_weights/default_name.pth', omni_vec: bool = False,
          vit: bool = False, need_softmax: bool = False, patchtst: bool = False, patchtsmixer: bool = False,
          model_input_type: str = 'singular') -> Dict[
    str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and val_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, logger.infos and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              val_loss: [...],
              val_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              val_loss: [1.2641, 1.5706],
              val_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "val_loss": [],
               }

    # Make sure that model is on target device
    model.to(device)

    best_val_loss = 10 ** 6
    patience_count = 0

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                device=device, lr_scheduling=lr_scheduling, omni_vec=omni_vec, vit=vit,
                                need_softmax=need_softmax, patchtst=patchtst, patchtsmixer=patchtsmixer, logger=logger,
                                model_input_type=model_input_type)
        val_loss = test_step(model=model,
                             dataloader=test_dataloader,
                             loss_fn=loss_fn,
                             device=device, vit=vit, need_softmax=need_softmax, patchtst=patchtst,
                             patchtsmixer=patchtsmixer, model_input_type=model_input_type)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            if save_best:
                torch.save(model.state_dict(), save_path + '.pth')
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(
                    f"Epoch: {epoch + 1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"val_loss: {val_loss:.4f} | "
                )
                return results

        # logger.info out what's happening
        logger.info(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

    # Return the filled results at the end of the epochs
    return results


def k_fold_train(model: torch.nn.Module,
                 data: Dict[Union[str, int], np.ndarray], df: pd.DataFrame, expected_shape: Iterable[int],
                 n_folds: int,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 epochs: int,
                 batch_size: int,
                 device: Union[torch.device, str], logger: logging.Logger, lr_scheduling: bool = False,
                 patience: int = 10000,
                 save_best: bool = True, save_path: str = 'model_weights/default_name.pth', omni_vec: bool = False,
                 vit: bool = False, patchtst: bool = False,
                 init_dict_path: str = f'model_weights/{uuid.uuid1()}.pth',
                 patchtsmixer: bool = False, need_softmax: bool = False, model_input_type: str = 'singular',
                 augment: bool = True, id_dtype: object = str, one_cycle: bool = True):
    torch.save(model.state_dict(), init_dict_path)
    folder = GroupKFold(n_splits=n_folds)
    folds = folder.split(df, df.target, df.patient_id)
    fold_results = {}
    base_save_path = save_path.split('.pth')[0]
    lr_optimizer = optimizer
    for num, (train_data, test_data) in enumerate(folds):
        df_train = df.loc[train_data]
        df_val = df.loc[test_data]
        train_dataset = GenericData(df=df_train, data_dict=data, expected_shape=expected_shape, augment=augment,
                                    id_dtype=id_dtype)
        test_dataset = GenericData(df=df_val, data_dict=data, expected_shape=expected_shape, augment=False,
                                   id_dtype=id_dtype)
        fold_save_path = base_save_path + f'_fold_{num}.pth'
        logger.info(f'Starting training of fold {num}')
        train_loader, test_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True), DataLoader(
            test_dataset, batch_size=batch_size)
        if one_cycle:
            lr_optimizer = OneCycleLR(optimizer=optimizer,
                                      max_lr=1e-3,
                                      epochs=epochs,
                                      steps_per_epoch=len(train_loader),
                                      pct_start=0.1,
                                      anneal_strategy="cos",
                                      final_div_factor=100,
                                      )
        fold_results[f'fold_{num}'] = train(model=model, train_dataloader=train_loader, test_dataloader=test_loader,
                                            optimizer=lr_optimizer, loss_fn=loss_fn,
                                            lr_scheduling=lr_scheduling, epochs=epochs,
                                            device=device, patience=patience, save_path=fold_save_path,
                                            save_best=save_best, vit=vit, omni_vec=omni_vec, patchtsmixer=patchtsmixer,
                                            patchtst=patchtst, need_softmax=need_softmax, logger=logger,
                                            model_input_type=model_input_type)
        model.load_state_dict(torch.load(init_dict_path))
    return fold_results


def two_staged_k_fold_train(model: torch.nn.Module,
                            first_stage_dataset: torch.utils.data.Dataset,
                            second_stage_dataset: torch.utils.data.Dataset,
                            df_second_stage: pd.DataFrame,
                            n_folds: int,
                            optimizer: torch.optim.Optimizer,
                            loss_fn: torch.nn.Module,
                            epochs: int,
                            batch_size: int,
                            device: Union[torch.device, str], logger: logging.Logger, lr_scheduling: bool = False,
                            patience: int = 10000,
                            save_best: bool = True, save_path: str = 'model_weights/default_name.pth',
                            omni_vec: bool = False,
                            vit: bool = False, patchtst: bool = False,
                            init_dict_path: str = f'model_weights/model_init_dict_{uuid.uuid1()}.pth',
                            patchtsmixer: bool = False, need_softmax: bool = False, model_input_type: str = 'singular'):
    torch.save(model.state_dict(), init_dict_path)
    folder = GroupKFold(n_splits=n_folds)
    folds = folder.split(df_second_stage, df_second_stage.target, df_second_stage.patient_id)
    fold_results = {}
    base_save_path = save_path.split('.pth')[0]
    for num, (train_data, test_data) in enumerate(folds):
        fold_save_path = base_save_path + f'_fold_{num}.pth'
        logger.info(f'Starting first stage training of fold {num}')

        # First Stage training
        train_loader, test_loader = DataLoader(first_stage_dataset, batch_size=batch_size), DataLoader(
            second_stage_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_data))
        fold_results[f'first_stage_fold_{num}'] = train(model=model, train_dataloader=train_loader,
                                                        test_dataloader=test_loader,
                                                        optimizer=optimizer, loss_fn=loss_fn,
                                                        lr_scheduling=lr_scheduling, epochs=epochs,
                                                        device=device, patience=patience, save_path=fold_save_path,
                                                        save_best=save_best, vit=vit, omni_vec=omni_vec,
                                                        patchtsmixer=patchtsmixer,
                                                        patchtst=patchtst, need_softmax=need_softmax, logger=logger,
                                                        model_input_type=model_input_type)

        # Second stage training
        train_loader, test_loader = DataLoader(second_stage_dataset, batch_size=batch_size,
                                               sampler=torch.utils.data.SubsetRandomSampler(train_data)), DataLoader(
            second_stage_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_data))
        fold_results[f'second_stage_fold_{num}'] = train(model=model, train_dataloader=train_loader,
                                                         test_dataloader=test_loader,
                                                         optimizer=optimizer, loss_fn=loss_fn,
                                                         lr_scheduling=lr_scheduling, epochs=epochs,
                                                         device=device, patience=patience, save_path=fold_save_path,
                                                         save_best=save_best, vit=vit, omni_vec=omni_vec,
                                                         patchtsmixer=patchtsmixer,
                                                         patchtst=patchtst, need_softmax=need_softmax, logger=logger,
                                                         model_input_type=model_input_type)
        # Reset model by loading freshly initialized state dict
        model.load_state_dict(torch.load(init_dict_path))
    return fold_results


def two_staged_train(model: torch.nn.Module,
                     first_stage_train_dataloader: torch.utils.data.DataLoader,
                     second_stage_train_dataloader: torch.utils.data.DataLoader,
                     test_dataloader: torch.utils.data.DataLoader,
                     optimizer: torch.optim.Optimizer,
                     loss_fn: torch.nn.Module,
                     epochs: int,
                     logger: logging.Logger,
                     device: Union[torch.device, str], lr_scheduling: bool = False,
                     patience: int = 10000,
                     save_best: bool = True, save_path: str = 'model_weights/default_name.pth', omni_vec: bool = False,
                     vit: bool = False, need_softmax: bool = False, patchtst: bool = False,
                     patchtsmixer: bool = False) -> Tuple[Dict[str, List], Dict[str, List]]:
    results_stage_one = train(model=model, train_dataloader=first_stage_train_dataloader,
                              test_dataloader=test_dataloader,
                              optimizer=optimizer, loss_fn=loss_fn,
                              lr_scheduling=lr_scheduling, epochs=epochs,
                              device=device, patience=patience, save_path=save_path,
                              save_best=save_best, vit=vit, omni_vec=omni_vec, patchtsmixer=patchtsmixer,
                              patchtst=patchtst, need_softmax=need_softmax, logger=logger)
    results_stage_two = train(model=model, train_dataloader=second_stage_train_dataloader,
                              test_dataloader=test_dataloader,
                              optimizer=optimizer, loss_fn=loss_fn,
                              lr_scheduling=lr_scheduling, epochs=epochs,
                              device=device, patience=patience, save_path=save_path,
                              save_best=save_best, vit=vit, omni_vec=omni_vec, patchtsmixer=patchtsmixer,
                              patchtst=patchtst, need_softmax=need_softmax, logger=logger)
    return results_stage_one, results_stage_two
