import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
import time
import csv
import copy
import os
from tqdm import tqdm
import numpy as np
from deemodel import prepare_model
from trygen import NepalDataset, NepalDataGenerator
from pan import PAN
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassJaccardIndex
# from torchmetrics import IoU
import torch.nn.functional as F



# def compute_iou(preds, labels, threshold=0.5):
#     preds = torch.sigmoid(preds)
#     preds = (preds > threshold).float()
#     preds_bool = preds.bool()
#     labels_bool = labels.bool()
#     intersection = (preds_bool & labels_bool).float().sum((1, 2, 3))
#     union = (preds_bool | labels_bool).float().sum((1, 2, 3))
#     iou = (intersection + 1e-6) / (union + 1e-6)
#     return iou.mean()


# def compute_iou(preds, labels, threshold=0.5):
#     # Apply sigmoid to the predictions and threshold to get binary masks
#     preds = torch.sigmoid(preds)
#     preds = (preds > threshold).float()
    
#     # Initialize IoU for each channel
#     iou_per_channel = []
    
#     # Compute IoU for each channel separately
#     for channel in range(preds.shape[1]):
#         preds_bool = preds[:, channel].bool()
#         labels_bool = labels[:, channel].bool()
        
#         intersection = (preds_bool & labels_bool).float().sum((1, 2))
#         union = (preds_bool | labels_bool).float().sum((1, 2))
        
#         iou = (intersection + 1e-6) / (union + 1e-6)
#         iou_per_channel.append(iou)
    
#     # Stack the IoUs and compute the mean across channels and batches
#     iou_per_channel = torch.stack(iou_per_channel, dim=1) 
#     return iou_per_channel.mean()
# def compute_iou(preds, labels, num_classes=2, threshold=0.5):
#     # Determine the device from the input tensors
#     device = preds.device
    
#     # Apply sigmoid to the predictions and threshold to get binary masks
#     preds = torch.softmax(preds)
#     preds = (preds > threshold).float()
    
#     # Check if predictions and labels have the expected shape
#     if preds.shape != labels.shape:
#         raise ValueError(f"Shape mismatch: preds shape {preds.shape} and labels shape {labels.shape} should be the same.")
    
#     # Initialize the Jaccard Index metric and move it to the correct device
#     jaccard = MulticlassJaccardIndex(num_classes=num_classes, average='macro').to(device)
    
#     # Compute the Jaccard Index (IoU) for each batch
#     iou = jaccard(preds, labels.int())
    
#     return iou

# def compute_iou(preds, labels, num_classes=3, thresold = 0.5):
#     preds = preds.to("cpu")
#     labels = labels.to("cpu")
#     print("preds",len(np.unique(preds.detach().numpy().flatten())))
#     print("labels", np.unique(labels.detach().numpy()))
#     # Convert numpy arrays to PyTorch tensors
#     preds = torch.tensor(preds, dtype=torch.float32)
#     labels = torch.tensor(labels, dtype=torch.float32)

#     # # Convert one-hot encoded masks to integer-encoded class labels
#     preds = torch.argmax(preds, dim=-1)
#     labels = torch.argmax(labels, dim=-1)
#     # print(preds)
#     # print(labels)

#     jaccard =  MulticlassJaccardIndex(num_classes=num_classes)
#     iou_scores = jaccard(preds, labels)
#     return iou_scores

# def compute_iou(preds, labels, threshold=0.5):
#     preds = torch.sigmoid(preds)
#     preds = (preds > threshold).float()
    
#     # For unlabelled
#     preds_unlabelled = preds[:, 0, :, :]
#     labels_unlabelled = labels[:, 0, :, :]
    
#     # For buildings
#     preds_buildings = preds[:, 1, :, :]
#     labels_buildings = labels[:, 1, :, :]

#     # For woodland
#     preds_woodland = preds[:, 2, :, :]
#     labels_woodland = labels[:, 2, :, :]

#     intersection_unlabelled = (preds_unlabelled * labels_unlabelled).sum((1, 2))
#     union_unlabelled = (preds_unlabelled + labels_unlabelled).sum((1, 2)) - intersection_unlabelled

#     intersection_buildings = (preds_buildings * labels_buildings).sum((1, 2))
#     union_buildings = (preds_buildings + labels_buildings).sum((1, 2)) - intersection_buildings
    
#     intersection_woodland = (preds_woodland * labels_woodland).sum((1, 2))
#     union_woodland = (preds_woodland + labels_woodland).sum((1, 2)) - intersection_woodland
    
#     iou_buildings = (intersection_buildings + 1e-6) / (union_buildings + 1e-6)
#     iou_woodland = (intersection_woodland + 1e-6) / (union_woodland + 1e-6)
#     iou_unlabelled = (intersection_unlabelled + 1e-6)/(union_unlabelled+1e-6)
    
#     iou_mean = (iou_buildings.mean() + iou_woodland.mean() + iou_unlabelled.mean()) / 3.0
    
#     return iou_mean

def compute_iou(preds, labels, threshold=0.5, epsilon=torch.finfo(torch.float).eps):
    """
    Calculates the mean Intersection-over-Union (IoU) for semantic segmentation.

    Args:
        preds (torch.Tensor): Predicted segmentation masks (probabilities before thresholding).
        labels (torch.Tensor): Ground truth segmentation labels.
        threshold (float, optional): Threshold for binary classification (default: 0.5).
        epsilon (float, optional): Small value to avoid division by zero (default: torch.finfo(torch.float).eps).

    Returns:
        float: Mean IoU across all classes.
    """

    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    # Separate predictions and labels for each class
    n_classes = preds.shape[1]  # Assuming channels represent class probabilities
    iou_per_class = []
    for i in range(n_classes):
        intersection = (preds[:, i, :, :] * labels[:, i, :, :]).sum((1, 2))
        union = (preds[:, i, :, :] + labels[:, i, :, :]).sum((1, 2)) - intersection

        # Handle potential division by zero with epsilon
        iou = (intersection + epsilon) / (union + epsilon)
        iou_per_class.append(iou.mean())

    iou_mean = sum(iou_per_class) / n_classes
    return iou_mean



# def compute_iou(preds, labels, num_classes=3, threshold=0.5):
#     # Determine the device from the input tensors
#     device = preds.device
    
#     # Apply sigmoid to the predictions and threshold to get binary masks
#     preds = torch.sigmoid(preds)
#     preds = (preds > threshold).float()
    
#     # Check if predictions and labels have the expected shape
#     if preds.shape != labels.shape:
#         raise ValueError(f"Shape mismatch: preds shape {preds.shape} and labels shape {labels.shape} should be the same.")
    
#     # Extract building and woodland masks from labels
#     unlabelled = (labels == 1).float()
#     building_mask = (labels == 2).float()
#     woodland_mask = (labels == 3).float()
    
#     # Combine masks to get background mask
#     background_mask = 1 - building_mask - woodland_mask
    
#     # Stack masks along channel dimension
#     labels = torch.stack([unlabelled, building_mask, woodland_mask], dim=1)
    
#     # Initialize the IoU metric
#     iou = IoU(num_classes=num_classes).to(device)
    
#     # Compute the IoU for each batch
#     iou_score = iou(preds, labels)
    
#     return iou_score

import torch

# def compute_iou(preds, labels, threshold=0.5):
#     preds = torch.sigmoid(preds)
#     preds = (preds > threshold).float()
    
#     # For buildings
#     preds_buildings = preds[:, 0, :, :]
#     labels_buildings = labels[:, 0, :, :]
    
#     # For woodland
#     preds_woodland = preds[:, 1, :, :]
#     labels_woodland = labels[:, 1, :, :]
    
#     intersection_buildings = (preds_buildings * labels_buildings).sum((1, 2))
#     union_buildings = (preds_buildings + labels_buildings).sum((1, 2)) - intersection_buildings
    
#     intersection_woodland = (preds_woodland * labels_woodland).sum((1, 2))
#     union_woodland = (preds_woodland + labels_woodland).sum((1, 2)) - intersection_woodland
    
#     iou_buildings = (intersection_buildings + 1e-6) / (union_buildings + 1e-6)
#     iou_woodland = (intersection_woodland + 1e-6) / (union_woodland + 1e-6)
    
#     iou_mean = (iou_buildings.mean() + iou_woodland.mean()) / 2.0
    
#     return iou_mean

# def compute_iou(pred, label, num_classes=3):
#     pred = F.softmax(pred, dim=1) 
#     print("1",pred.shape)           
#     pred = torch.argmax(pred, dim=1).squeeze(1)
#     print("2",pred.shape)
#     iou_list = list()
#     present_iou_list = list()

#     pred = pred.view(-1)
#     print(pred.shape)
#     label = label.view(-1)
#     print(label.shape)
#     # Note: Following for loop goes from 0 to (num_classes-1)
#     # and ignore_index is num_classes, thus ignore_index is
#     # not considered in computation of IoU.
#     for sem_class in range(1, num_classes+1):
#         pred_inds = (pred == sem_class)
#         # print(pred_inds.shape)
#         target_inds = (label == sem_class)
#         # print(target_inds.shape)
#         if target_inds.long().sum().item() == 0:
#             iou_now = float('nan')
#         else: 
#             intersection_now = (pred_inds[target_inds]).long().sum().item()
#             union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
#             iou_now = float(intersection_now) / float(union_now)
#             present_iou_list.append(iou_now)
#         iou_list.append(iou_now)
#     return np.mean(present_iou_list)

def train_model(model, train_dataloader, val_dataloader, num_epochs=100, learning_rate=0.0001, patience=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    early_stopping_counter = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_iou = 0.0

        for images, masks in tqdm(train_dataloader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print(outputs.shape, masks.shape)
            iou = compute_iou(outputs, masks)
            running_iou += iou.item()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_iou = running_iou / len(train_dataloader)
        train_losses.append(epoch_loss)
        train_ious.append(epoch_iou)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train IoU: {epoch_iou:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_iou = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_dataloader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                val_loss = criterion(outputs, masks)
                val_running_loss += val_loss.item()
                val_iou = compute_iou(outputs, masks)
                val_running_iou += val_iou.item()

        val_epoch_loss = val_running_loss / len(val_dataloader)
        val_epoch_iou = val_running_iou / len(val_dataloader)
        val_ious.append(val_epoch_iou)
        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_epoch_loss:.4f}, Val IoU: {val_epoch_iou:.4f}")


        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'bestmodeloverall22.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

    model.load_state_dict(best_model_weights)
    print("Training completed.")

    for epoch in range(len(train_losses)):
        print(f"Epoch {epoch + 1}: Train Loss = {train_losses[epoch]:.4f}, Val Loss = {val_losses[epoch]:.4f}")


def train_model_with_deep():
    num_classes = 3
    deep_model = prepare_model()

    data_path = "./Output_mg/513x513/"
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = NepalDataset(data_path, transform=transform, training=True)
    val_dataset = NepalDataset(data_path, transform=transform, training=False)

    batch_size = 3
    shuffle = True
    train_dataloader = NepalDataGenerator(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = NepalDataGenerator(val_dataset, batch_size=batch_size, shuffle=False)

    train_model(deep_model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    train_model_with_deep()


