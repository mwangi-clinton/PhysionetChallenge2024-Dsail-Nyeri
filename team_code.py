#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
import random
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from PIL import Image
from collections import OrderedDict
from data_loader import get_training_and_validation_loaders
from functools import partial
from helper_code import *
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from typing import Callable, Optional
from tqdm import tqdm
import timm
from sklearn.metrics import f1_score



DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # For the first GPU
EPOCHS = 10
CLASSIFICATION_THRESHOLD=0.6
CLASSIFICATION_DISTANCE_TO_MAX_THRESHOLD=0.1
LIST_OF_ALL_LABELS=['NORM', 'Acute MI', 'Old MI', 'STTC', 'CD', 'HYP', 'PAC', 'PVC', 'AFIB/AFL', 'TACHY', 'BRADY'] 
RESIZE_TEST_IMAGES=(550, 711)
OPTIM_LR=1e-4
OPTIM_WEIGHT_DECAY=1e-5
SCHEDULER_STEP_SIZE=10
SCHEDULER_GAMMA=0.5

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.


class PhysionetCNN(nn.Module):
    def __init__(self, list_of_classes):
        super(PhysionetCNN, self).__init__()
        self.list_of_classes = list_of_classes
        self.num_classes = len(self.list_of_classes)
        # Load the pre-trained EfficientNetB5 model
        self.model = timm.create_model('efficientnet_b5', pretrained=False)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, self.num_classes),
            nn.Sigmoid()
        )

        # Freeze the first layers of the EfficientNetB5 backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers (optional: adjust as needed)
        for param in list(self.model[-1].parameters()):
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

    
def train_models(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the data...
    if verbose:
        print('Loading the data...')

    classification_images = list()  # list of image paths
    classification_labels = list()  # list of lists of strings

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        record_parent_folder = os.path.dirname(record)

        # Some images may not be labeled, so we'll exclude those
        labels = load_labels(record)
        if labels:
            # I'm imposing a further condition: the label strings should be nonempty
            nonempty_labels = [l for l in labels if l != '']
            if nonempty_labels:
                # Add the first image to the list
                images = get_image_files(record)
                classification_images.append(os.path.join(record_parent_folder, images[0]))
                classification_labels.append(nonempty_labels)

    # We expect some images to be labeled for classification.
    if not classification_labels:
        raise Exception('There are no labels for the data.')

    # Fix an ordering of the labels
    num_classes = len(LIST_OF_ALL_LABELS)

    # Train the models.
    if verbose:
        print('Training the models on the data...')

    # Classification task
    training_loader, validation_loader = get_training_and_validation_loaders(
        LIST_OF_ALL_LABELS, classification_images, classification_labels)

    # Initialize a model
    classification_model = PhysionetCNN(LIST_OF_ALL_LABELS).to(DEVICE)
    classification_model.load_state_dict(
        torch.load("base_model/classification_model.pth", weights_only=False))
    
    classification_model.train()

    for param in classification_model.parameters():  # fine tune all the layers
        param.requires_grad = True

    loss = nn.BCELoss()  # binary cross entropy loss for multilabel classification
    opt = optim.Adam(classification_model.parameters(), lr=OPTIM_LR, weight_decay=OPTIM_WEIGHT_DECAY)
    scheduler = StepLR(opt, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    N_loss = []
    N_loss_valid = []
    train_auprc = []
    valid_auprc = []
    train_auroc = []
    valid_auroc = []
    f1_train = []
    f1_valid = []

    plot_folder = os.path.join(model_folder, "training_figures")
    os.makedirs(plot_folder, exist_ok=True)

    # Filename to save the final weights to
    final_weights = None

    # Training Loop
    for epoch in tqdm(range(EPOCHS)):
        # Initialization of variables for plotting the progress
        N_item_sum = 0
        N_item_sum_valid = 0
        targets_train = []
        outputs_train = []
        targets_valid = []
        outputs_valid = []

        ### Training part
        if verbose:
            print(f"============================[{epoch}]============================")
        classification_model.train()
        for i, (image, label) in enumerate(training_loader):
            opt.zero_grad()

            image = image.float().to(DEVICE)
            label = label.to(torch.float).to(DEVICE)
            prediction = classification_model(image)

            # loss
            N = loss(prediction, label)
            N.backward()
            N_item = N.item()
            N_item_sum += N_item

            # gradient clipping plus optimizer
            torch.nn.utils.clip_grad_norm_(classification_model.parameters(), max_norm=10)
            opt.step()
            if verbose:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {N_item}")

            targets_train.append(label.data.cpu().numpy())
            outputs_train.append(prediction.data.cpu().numpy())

        ### Validation part
        classification_model.eval()
        with torch.no_grad():
            for j, (image, label) in enumerate(validation_loader):
                image = image.float().to(DEVICE)
                label = label.to(torch.float).to(DEVICE)
                prediction = classification_model(image)

                N = loss(prediction, label)
                N_item = N.item()
                N_item_sum_valid += N.item()

                targets_valid.append(label.data.cpu().numpy())
                outputs_valid.append(prediction.data.cpu().numpy())
                print(f"Epoch: {epoch}, Valid Iteration: {j}, Loss: {N_item}")

        scheduler.step()

        # Logging the outputs and targets to calculate metrics
        targets_train = np.concatenate(targets_train, axis=0).T
        outputs_train = np.concatenate(outputs_train, axis=0).T
        targets_valid = np.concatenate(targets_valid, axis=0).T
        outputs_valid = np.concatenate(outputs_valid, axis=0).T

        outputs_valid_pred = (outputs_valid > 0.5).astype(int)
        micro_f1 = f1_score(targets_valid, outputs_valid_pred, average='micro')
        macro_f1 = f1_score(targets_valid, outputs_valid_pred, average='macro')
        weighted_f1 = f1_score(targets_valid, outputs_valid_pred, average='weighted')

        print(f"Epoch: {epoch}, Micro F1: {micro_f1:.2f}, Macro F1: {macro_f1:.2f}, Weighted F1: {weighted_f1:.2f}")

        auprc_t = average_precision_score(y_true=targets_train, y_score=outputs_train)
        auroc_t = roc_auc_score(y_true=targets_train, y_score=outputs_train)
        auprc_v = average_precision_score(y_true=targets_valid, y_score=outputs_valid)
        auroc_v = roc_auc_score(y_true=targets_valid, y_score=outputs_valid)

        train_auprc.append(auprc_t)
        train_auroc.append(auroc_t)
        valid_auprc.append(auprc_v)
        valid_auroc.append(auroc_v)
        f1_train.append((micro_f1, macro_f1, weighted_f1))

        N_loss.append(N_item_sum / i)
        N_loss_valid.append(N_item_sum_valid / j)

        # saving loss function after each epoch so you can look at progress
        fig = plt.figure()
        plt.plot(N_loss, label="train")
        plt.plot(N_loss_valid, label="valid")
        plt.title("Loss function")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(plot_folder, "loss.png"))
        plt.close()

        fig = plt.figure()
        plt.plot(train_auprc, label="train auprc")
        plt.plot(valid_auprc, label="valid auprc")
        plt.plot(train_auroc, label="train auroc")
        plt.plot(valid_auroc, label="valid auroc")
        plt.title("AUPRC and AUROC")
        plt.xlabel('epoch')
        plt.ylabel('Performance')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(plot_folder, "auroc_auprc.png"))
        plt.close()

        # save model after each epoch
        file_path = os.path.join(model_folder, "model_weights_" + str(epoch) + ".pth")
        torch.save(classification_model.state_dict(), file_path)

        # If this is the last epoch, then the weights of the model will be saved to this file
        final_weights = file_path

    # Create a folder for the models if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the models.
    save_classification_model(model_folder, LIST_OF_ALL_LABELS, final_weights)

    if verbose:
        print('Done.')


# Load your trained models. This function is *required*. You should edit this
# function to add your code, but do *not* change the arguments of this
# function. If you do not train one of the models, then you can return None for
# the model.
def load_models(model_folder, verbose):
    digitization_model = None

    classes_filename = os.path.join(model_folder, 'classes.txt')
    classes = joblib.load(classes_filename)

    classification_model = PhysionetCNN(classes).to(DEVICE) # instantiate a new copy of the model
    classification_filename = os.path.join(model_folder, "classification_model.pth")
    classification_model.load_state_dict(torch.load(classification_filename,weights_only=True))

    return digitization_model, classification_model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):

    # Run the digitization model; if you did not train this model, then you can set signal = None.
    signal = None

    # Run the classification model.
    classes = classification_model.list_of_classes

    # Open the image:
    record_parent_folder=os.path.dirname(record)
    image_files=get_image_files(record)
    image_path=os.path.join(record_parent_folder, image_files[0])
    img = Image.open(image_path)
    # FIXME: repeated code---maybe factor out opening the image from a record
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # transform the image and make it suitable as input
    img = transforms.Resize(RESIZE_TEST_IMAGES)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
    img = img.unsqueeze(0)

    # send it to the GPU if necessary
    img = img.float().to(DEVICE)

    classification_model.eval()
    with torch.no_grad():
        probabilities = torch.squeeze(classification_model(img), 0).tolist()
        predictions=list()
        for i in range(len(classes)):
            if probabilities[i] >= CLASSIFICATION_THRESHOLD:
                predictions.append(classes[i])

    # backup if none is over the threshold: use the max
    if predictions==[]:
        highest_probability=max(probabilities)
        for i in range(len(classes)):
            if abs(highest_probability - probabilities[i]) <= CLASSIFICATION_DISTANCE_TO_MAX_THRESHOLD:
                predictions.append(classes[i])

    return signal, predictions

#########################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
#########################################################################################

# Extract features.
def extract_features(record):
    images = load_images(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

# Save your trained models.
def save_classification_model(model_folder,
                list_of_classes=None,
                final_weights=None):

    if final_weights is not None:
        classes=filename = os.path.join(model_folder, 'classes.txt')
        joblib.dump(list_of_classes, filename, protocol=0)

        # copy the file with the final weights to the model path
        model_filename=os.path.join(model_folder, "classification_model.pth")
        shutil.copyfile(final_weights, model_filename)
