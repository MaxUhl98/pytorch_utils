"""
Most Credit to https://github.com/mrdbourke/pytorch-deep-learning
I implemented these Functions while taking the ZTM Pytorch Deep Learning Course"""
import torch
from torch import nn
import time
from typing import List, Tuple, Dict
import PIL
from torchvision.transforms import Compose


def predict_and_store(model, samples: List[str], sample_classes: List[int], transforms: Compose):
    """Predicts Multiple samples via Filepath and returns the results in a list of dictionaries containing:
    'pred_correct','pred_probs', 'prediction', 'prediction_time' and their respective observed values"""
    # Set model to evaluation mode
    model.eval()
    # Use inference mode to avoid problems
    with model.inference_mode():
        # Get start timestamp
        t_total = time.time()
        # Create list to save
        results = []
        # iterate over samples
        for num, sample in enumerate(samples):
            sample_results = {}
            # Sample stattime
            t_sample = time.time()
            # open image
            img = PIL.Image.open(sample)
            # transform image
            img = transforms(img)
            # make prediction
            pred_probs = model(img)
            # save prediction probabilities
            sample_results['pred_probs'] = pred_probs
            # save predicted class
            sample_results['prediction'] = torch.argmax(pred_probs, dim=1)
            # calculate time taken
            time_taken = time.time() - t_sample
            # save time taken
            sample_results['prediction_time'] = time_taken
            # save the truth value of the prediction
            sample_results['pred_correct'] = sample_results['prediction'] == sample_classes[num]
            print(f'This Prediction took {time_taken:.5f} seconds and was {sample_results["pred_correct"]}')
            # Add dictionary to list of results
            results.append(sample_results)
        print(f'Total time taken for {len(samples)} Predictions were {time.time() - t_total:.5f} Seconds')
    return results


def predict(img, model: nn.Module) -> Tuple[Dict, float]:
    """Predicts one Image(in tensor form) and returns its class probabilities insude a dictionary and
     the time taken to predict in seconds as a float"""
    # Start timing
    t_s = time.time()
    # Model in Eval Mode
    model.eval()
    # Predict
    with model.inference_mode():
        pred_probs = model(img)
    # Save predictions
    pred_info = {'pizza': pred_probs[0].item(), 'steak': pred_probs[1].item(), 'sushi': pred_probs[2].item()}
    # Calculate Runtime
    pred_time = time.time() - t_s
    return pred_info, pred_time
