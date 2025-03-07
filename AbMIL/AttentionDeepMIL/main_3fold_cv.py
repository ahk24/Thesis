#!/usr/bin/env python3
from __future__ import print_function
import logging
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import os
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data import random_split, ConcatDataset, DataLoader

# Import your dataset and models
from dataloader import PreConcatenatedDataset
from model import Attention, GatedAttention

# Configure logging to display timestamps and log levels.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def train_one_epoch(model, loader, optimizer, device):
    logging.info("Starting training epoch")
    model.train()
    epoch_loss = 0.0
    epoch_error = 0.0

    for batch_idx, (bag, label) in enumerate(loader):
        logging.debug("Training batch %d - bag shape: %s", batch_idx, bag.shape)
        bag = bag.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        loss, _ = model.calculate_objective(bag, label)
        error, _ = model.calculate_classification_error(bag, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_error += error

    n = len(loader)
    avg_loss = epoch_loss / n if n > 0 else 0.0
    avg_error = epoch_error / n if n > 0 else 0.0
    logging.info("Finished training epoch: Avg Loss: %.4f, Avg Error: %.4f", avg_loss, avg_error)
    return avg_loss, avg_error

def evaluate(model, loader, device):
    logging.info("Starting evaluation")
    model.eval()
    epoch_loss = 0.0
    epoch_error = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (bag, label) in enumerate(loader):
            logging.debug("Evaluating batch %d - bag shape: %s", batch_idx, bag.shape)
            bag = bag.to(device)
            label = label.to(device)
            loss, _ = model.calculate_objective(bag, label)
            error, _ = model.calculate_classification_error(bag, label)

            epoch_loss += loss.item()
            epoch_error += error

            # Predict probabilities
            Y_prob, _, _ = model(bag)
            prob = Y_prob.mean().item()  # average if multiple branches

            all_labels.append(label.item())
            all_probs.append(prob)

    n = len(loader)
    avg_loss = epoch_loss / n if n else 0.0
    avg_error = epoch_error / n if n else 0.0
    logging.info("Finished evaluation: Avg Loss: %.4f, Avg Error: %.4f", avg_loss, avg_error)
    return avg_loss, avg_error, all_labels, all_probs

def main():
    logging.info("Script started")
    
    # Hardcoded configuration
    use_cuda = False
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    data_root = "/data/temporary/projects/mutation-prediction/csvs/3-folds-prostate-bladder-breast"
    concatenated_data_dir = "/data/temporary/amirhosein/model_script_concatenated"  # Directory with pre-concatenated data

    model_type = "attention"  # or "gated_attention"
    input_dim = 1024
    hidden_dim = 497
    attn_dim = 255
    attn_branches = 1
    epochs = 10
    lr = 0.00025128332628065683
    weight_decay = 1.014746907346501e-05
    patience = 4  # early stopping patience

    fold_metrics = {'precision': [], 'recall': [], 'auc': []}

    # 3-fold cross-validation loop
    for fold_idx in range(0, 3):
        logging.info("=== Starting FOLD %d ===", fold_idx)
        fold_subdir = f"fold-{fold_idx}"
        fold_path = os.path.join(data_root, fold_subdir)

        train_csv = os.path.join(fold_path, "train.csv")
        val_csv   = os.path.join(fold_path, "tune.csv")
        test_csv  = os.path.join(fold_path, "test.csv")

        logging.info("Loading datasets for fold %d", fold_idx)
        # Create datasets using pre-concatenated files
        train_dataset = PreConcatenatedDataset(concatenated_data_dir, train_csv)
        val_dataset   = PreConcatenatedDataset(concatenated_data_dir, val_csv)
        test_dataset  = PreConcatenatedDataset(concatenated_data_dir, test_csv)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

        # Initialize model
        logging.info("Initializing model for fold %d", fold_idx)
        if model_type == "attention":
            model = Attention(input_dim=input_dim, M=hidden_dim, L=attn_dim, attention_branches=attn_branches)
        else:
            model = GatedAttention(input_dim=input_dim, M=hidden_dim, L=attn_dim, attention_branches=attn_branches)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Training and validation loop with early stopping
        best_val_loss = float("inf")
        best_state_dict = None
        no_improvement_counter = 0

        for epoch_idx in range(1, epochs + 1):
            logging.info("[Fold %d | Epoch %d] Starting training", fold_idx, epoch_idx)
            train_loss, train_error = train_one_epoch(model, train_loader, optimizer, device)
            logging.info("[Fold %d | Epoch %d] Training complete. Loss: %.4f, Error: %.4f", fold_idx, epoch_idx, train_loss, train_error)
            
            logging.info("[Fold %d | Epoch %d] Starting validation", fold_idx, epoch_idx)
            val_loss, val_error, _, _ = evaluate(model, val_loader, device)
            logging.info("[Fold %d | Epoch %d] Validation complete. Loss: %.4f, Error: %.4f", fold_idx, epoch_idx, val_loss, val_error)

            # Check if current epoch improved validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = model.state_dict()
                no_improvement_counter = 0
                logging.info("[Fold %d | Epoch %d] New best model found (Validation Loss: %.4f)", fold_idx, epoch_idx, best_val_loss)
            else:
                no_improvement_counter += 1
                logging.info("[Fold %d | Epoch %d] No improvement. Counter: %d", fold_idx, epoch_idx, no_improvement_counter)

            # If no improvement for "patience" epochs, break early
            if no_improvement_counter >= patience:
                logging.info("[Fold %d] Early stopping triggered after %d epochs with no improvement.", fold_idx, patience)
                break

        # Load best model for testing
        if best_state_dict:
            model.load_state_dict(best_state_dict)
            logging.info("Best model loaded for fold %d", fold_idx)

        # Testing
        logging.info("Starting testing for fold %d", fold_idx)
        test_loss, test_error, y_true, y_scores = evaluate(model, test_loader, device)
        logging.info("[Fold %d] Testing complete. Loss: %.4f, Error: %.4f", fold_idx, test_loss, test_error)

        # Compute metrics
        y_pred = [1 if p >= 0.5 else 0 for p in y_scores]
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        try:
            auc_val = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc_val = float('nan')

        logging.info("[Fold %d] Metrics - Precision: %.4f, Recall: %.4f, AUC: %.4f", fold_idx, precision, recall, auc_val)

        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['auc'].append(auc_val)

    # After 3 folds, compute and log average metrics
    avg_precision = sum(fold_metrics['precision']) / 3
    avg_recall    = sum(fold_metrics['recall'])    / 3
    valid_aucs    = [a for a in fold_metrics['auc'] if not (isinstance(a, float) and (a != a))]
    avg_auc       = sum(valid_aucs) / len(valid_aucs) if valid_aucs else float('nan')

    logging.info("=== 3-FOLD CROSS-VALIDATION RESULTS ===")
    logging.info("Average Precision: %.4f", avg_precision)
    logging.info("Average Recall:    %.4f", avg_recall)
    logging.info("Average AUC:       %.4f", avg_auc)
    logging.info("Script finished successfully")

if __name__ == "__main__":
    main()