"""
Multi-Task MIL with Contrastive Loss + Bayesian Optimization + 3-Fold CV using Lazy Loading and Mixed Precision Training

This script implements a multi-task MIL model that uses pre-computed embeddings 
(for example, tensors loaded from .pt files) for binary prediction of a target cancer while leveraging auxiliary data 
(cancer type) from other cancers. The auxiliary cancer type is provided as a string (e.g. "prostate", "breast", etc.). 
A global mapping is computed to convert these strings into integer labels, and samples with unknown or missing 
cancer types are discarded.

Key features:
  - 3-fold cross validation. Each fold (fold-0, fold-1, fold-2) contains CSV files for train, tune (validation) and test labels.
  - Modular data loading: the CSV files contain at least:
      • a case_id (used to match embedding filenames)
      • a primary (binary) label (default column name: "label")
      • an auxiliary label (cancer type as a string; default column name: "cancer_type")
  - Lazy loading is used: the dataset stores embedding file paths (instead of loading entire embeddings upfront) and loads them on demand.
  - Multi-task model with:
      • Shared feature extraction (an MLP) applied to each embedding.
      • An attention (or gated attention) mechanism to aggregate patch features into a bag representation.
      • A primary head for binary classification and an auxiliary head for cancer type classification.
      • A supervised contrastive loss computed on the bag representations.
  - Bayesian optimization (Optuna with multi-fidelity pruning) is used to tune hyperparameters.
  - Mixed precision training is enabled via torch.cuda.amp.
  - Extensive logging via the logging module and wandb integration.
  - ROC curve plotting and ROC-AUC measurement via scikit-learn and matplotlib.

Requirements:
  - PyTorch, pandas, numpy
  - optuna, wandb
  - scikit-learn, matplotlib
"""

import argparse
import os
import sys
import glob
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import wandb
import optuna
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.cuda.amp import autocast
from torch.amp import GradScaler  # For mixed precision training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import pickle



best_auc_global = 0.0         # tracks best avg-test AUC achieved so far
best_model_path = "/data/temporary/amirhosein/UNI_processed/best_model_6cancer_baseline_DDP.pth"
# -------------------------
# Setup Logging
# -------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------
# Data Loading Utilities (Lazy Loading)
# -------------------------
def compute_aux_mapping(fold_dir, aux_label_col):
    """
    Computes a global auxiliary label mapping dictionary from all CSV files across folds.
    Reads all CSVs from fold-0, fold-1, and fold-2 to collect valid cancer type strings 
    (converted to lower-case) and assigns a unique integer to each.
    Samples with "unknown", empty, or "nan" values are discarded.
    """
    aux_values = set()
    selected_folds = [int(f) for f in args.folds.split(",")]
    for fold in selected_folds:
        for csv_name in ["train_contrastive_updated.csv", "tune_contrastive_updated.csv", "test_contrastive_updated.csv"]:
            csv_path = os.path.join(fold_dir, f"fold-{fold}", csv_name)
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            valid_aux = df[aux_label_col].astype(str).str.strip().str.lower()
            valid_aux = valid_aux[~valid_aux.isin(["unknown"])]
            aux_values.update(valid_aux.unique())
    mapping = {val: idx for idx, val in enumerate(sorted(aux_values))}
    logger.info("Computed auxiliary mapping:")
    for key, value in mapping.items():
        logger.info(f"    {key}: {value}")
    return mapping

def build_dataset_info(label_file, embeddings_dir, label_col="TP53", case_id_col="case_id", 
                       aux_label_col=None, aux_mapping=None):
    """
    Reads a CSV file (without loading the embeddings) to build a list of tuples.
    Each tuple contains:
       (embedding_file_path, primary_label)  
       OR  
       (embedding_file_path, primary_label, aux_label)
    based on whether aux_label_col is provided.
    Samples with missing embeddings or invalid auxiliary labels are discarded.
    
    Additionally, if aux_label_col is provided, logs the distribution of auxiliary labels.
    """
    logger.info(f"Loading label info from: {label_file}")
    df = pd.read_csv(label_file)
    info_list = []
    skipped_due_to_aux = 0
    for idx, row in df.iterrows():
        case_id = str(row[case_id_col])
        primary_label = row[label_col]
        if aux_label_col is not None:
            aux_raw = row[aux_label_col]
            aux_str = str(aux_raw).strip().lower()
            #if aux_str in ["unknown"]:
            #    skipped_due_to_aux += 1
            #    continue
            if aux_mapping is not None:
                if aux_str not in aux_mapping:
                    logger.warning(f"Auxiliary label '{aux_str}' not in mapping; skipping case_id {case_id}.")
                    continue
                aux_label = aux_mapping[aux_str]
            else:
                Print("No auxiliary mapping provided. Using raw label.")
                try:
                    aux_label = int(aux_raw)
                except ValueError:
                    logger.warning(f"Invalid auxiliary label '{aux_raw}' for case_id {case_id}; skipping.")
                    skipped_due_to_aux += 1
                    continue
        pattern = os.path.join(embeddings_dir, f"*{case_id}*.pt")
        files = glob.glob(pattern)
        if not files:
            logger.warning(f"No embedding file found for case_id: {case_id}. Skipping sample.")
            continue
        file_path = files[0]
        if aux_label_col is not None:
            info_list.append((file_path, primary_label, int(aux_label)))
        else:
            info_list.append((file_path, primary_label))
    logger.info(f"Built dataset info for {len(info_list)} samples from {label_file}. Skipped {skipped_due_to_aux} samples due to invalid auxiliary labels.")

    # If auxiliary labels are available, log their distribution.
    if aux_label_col is not None and len(info_list) > 0:
        aux_labels = [sample[2] for sample in info_list]
        aux_counts = Counter(aux_labels)
        # Optionally, convert back to the original cancer type strings using an inverse mapping:
        inv_mapping = {v: k for k, v in aux_mapping.items()} if aux_mapping is not None else {}
        distribution_str = ", ".join([f"{inv_mapping.get(label, label)}: {count}" for label, count in aux_counts.items()])
        logger.info(f"Auxiliary label distribution for {label_file}: {distribution_str}")

    return info_list

class LazyEmbeddingsDataset(data.Dataset):
    """
    PyTorch Dataset for MIL using lazy loading.
    Instead of loading the embedding tensor at initialization,
    only the file path (and associated labels) is stored.
    The embedding is loaded from disk on each __getitem__ call.
    Each sample is a tuple:
         (embedding, primary_label, aux_label)  OR
         (embedding, primary_label)
    """
    def __init__(self, info_list):
        self.info_list = info_list

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, index):
        # Retrieve the stored file path and labels
        sample = self.info_list[index]
        file_path = sample[0]
        try:
            embedding = torch.load(file_path)
        except Exception as e:
            logger.warning(f"Error loading embedding from {file_path}: {e}")
            # In the event of error, return a dummy tensor (adjust as appropriate)
            embedding = torch.zeros((1,))  
        if len(sample) == 3:
            return embedding, sample[1], sample[2]
        else:
            return embedding, sample[1]


class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, info_list, dat_path, index_path):
        """
        info_list: [(file_path, primary_label, aux_label), …]
        dat_path:   path to all_embs.dat
        index_path: path to offsets.pkl
        """
        self.info = info_list
        # open memmap read-only; OS will page-cache as needed
        self.mm   = np.memmap(dat_path, mode="r", dtype="float32")
        # load offsets {file_path: (start, end)}
        with open(index_path, "rb") as f:
            self.offsets = pickle.load(f)

        # derive D from the first entry
        total = self.mm.shape[0]
        # note: you can reshape mm to (total, D) if you saved shape
        # but here we assume you know D or store it elsewhere.

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        path, prim_lbl, aux_lbl = self.info[idx]
        start, end = self.offsets[path]
        # slice out the right rows, then reshape to (n_patches, D)
        # you need to know D; e.g. hard-code or save it alongside index.pkl
        D = ...  # e.g. 1024 or load from a small metadata file
        emb = self.mm[start*D : end*D].reshape(end-start, D)
        emb = torch.from_numpy(emb)
        return emb, prim_lbl, aux_lbl

def collate_fn(batch):
    """
    Filter out any None items. Usually not needed if you skip them beforehand,
    but this is an extra safeguard in case something still yields None.
    """
    valid_items = []
    for item in batch:
        if item is not None:
            valid_items.append(item)
        else:
            logging.warning("Skipped a None item during collate_fn.")
    return valid_items


# -------------------------
# Supervised Contrastive Loss
# -------------------------
def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    Compute a supervised contrastive loss given a batch of features and integer labels.
    features: tensor of shape [batch_size, feature_dim]
    labels: tensor of shape [batch_size] of integers.
    Encourages features from samples with the same label to cluster together, 
    while pushing apart features with different labels.
    """
    if features.size(0) < 2:
        return torch.tensor(0.0, device=features.device)
    
    features = nn.functional.normalize(features, p=2, dim=1)
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    batch_size = features.size(0)
    mask = torch.eye(batch_size, device=features.device).bool()
    
    loss = 0.0
    count = 0
    for i in range(batch_size):
        pos_mask = (labels == labels[i])
        pos_mask[i] = False
        if pos_mask.sum().item() == 0:
            continue
        exp_sim = torch.exp(similarity_matrix[i][~mask[i]])
        denominator = exp_sim.sum()
        numerator = torch.exp(similarity_matrix[i][pos_mask]).sum()
        loss_i = -torch.log((numerator + 1e-8) / (denominator + 1e-8))
        loss += loss_i
        count += 1
    return loss / count if count > 0 else torch.tensor(0.0, device=features.device)

# -------------------------
# Model Definition: Multi-Task MIL with Contrastive Loss
# -------------------------
class MultiTaskMILClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, M=500, L=128, 
                 attention_type="attention", aux_num_classes=3):
        """
        Multi-task MIL model.
        Each bag is processed by:
          • A shared MLP feature extractor (applied patchwise).
          • A projection to an M-dimensional space.
          • An attention (or gated attention) mechanism to aggregate patch features into a bag-level representation.
        Two classifier heads are used:
          • A primary head for binary classification.
          • An auxiliary head for cancer type classification.
        The bag representation is also used to compute a supervised contrastive loss.
        """
        super(MultiTaskMILClassifier, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, M),
            nn.ReLU()
        )
        self.M = M
        self.L = L
        self.attention_type = attention_type
        
        if attention_type == "attention":
            self.attention_layer = nn.Sequential(
                nn.Linear(M, L),
                nn.Tanh(),
                nn.Linear(L, 1)
            )
        elif attention_type == "gated_attention":
            self.attention_V = nn.Sequential(
                nn.Linear(M, L),
                nn.Tanh()
            )
            self.attention_U = nn.Sequential(
                nn.Linear(M, L),
                nn.Sigmoid()
            )
            self.attention_layer = nn.Linear(L, 1)
        else:
            raise ValueError("Invalid attention_type. Choose 'attention' or 'gated_attention'.")
        
        self.classifier_head = nn.Linear(M, 1)   # Primary (binary) classifier head
        self.classifier_aux = nn.Linear(M, aux_num_classes)  # Auxiliary classifier head
        
    def forward(self, x):
        """
        Forward pass for a single bag.
        x: tensor of shape [N, input_dim] (where N is the number of patches).
        Returns:
          bag_rep: Aggregated bag-level representation (1 x M) then squeezed.
          binary_logit: Primary output (scalar logit).
          aux_logits: Auxiliary output (logits for cancer type classification).
          attention_weights: Attention weights (1 x N).
        """
        H = self.feature_extractor(x)        # [N, hidden_dim]
        H_proj = self.projection(H)            # [N, M]
        
        if self.attention_type == "attention":
            A = self.attention_layer(H_proj)   # [N, 1]
        else:
            A_V = self.attention_V(H_proj)       # [N, L]
            A_U = self.attention_U(H_proj)       # [N, L]
            A = self.attention_layer(A_V * A_U)  # [N, 1]
        
        A = torch.transpose(A, 0, 1)           # [1, N]
        A = torch.softmax(A, dim=1)            # [1, N]
        bag_rep = torch.mm(A, H_proj)          # [1, M]
        
        binary_logit = self.classifier_head(bag_rep)  # [1, 1]
        aux_logits = self.classifier_aux(bag_rep)       # [1, aux_num_classes]
        
        binary_logit = torch.squeeze(binary_logit)  # scalar
        bag_rep = torch.squeeze(bag_rep)            # [M]
        
        return bag_rep, binary_logit, aux_logits, A

# -------------------------
# Training and Evaluation Functions
# -------------------------

def evaluate_with_loss(model, data_loader, device,
                       primary_loss_fn, aux_loss_fn,
                       primary_weight=1.0, aux_weight=1.0):
    """
    Runs one pass over data_loader, returning:
      - val_loss: average weighted loss over all batches
      - val_auc:  ROC-AUC on the primary task
    """
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            batch_loss = 0.0
            for sample in batch:
                embedding, primary_label, aux_label = sample
                embedding = embedding.to(device).float()
                _, logit, aux_logits, _ = model(embedding)

                # Compute per-sample losses
                pl = primary_loss_fn(logit, torch.tensor([primary_label], device=device))
                al = aux_loss_fn(aux_logits, torch.tensor([aux_label], device=device))
                batch_loss += primary_weight * pl + aux_weight * al

                prob = torch.sigmoid(logit).item()
                all_probs.append(prob)
                all_labels.append(float(primary_label))

            running_loss += batch_loss.item()
            num_batches += 1

    val_loss = running_loss / num_batches if num_batches else 0.0
    val_auc  = roc_auc_score(all_labels, all_probs) if all_labels else 0.0
    return val_loss, val_auc


def plot_roc(all_labels, all_probs, output_path):
    """
    Plot the ROC curve using matplotlib and save the figure.
    """
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.savefig(output_path)
    plt.close()
    logger.info(f"ROC curve saved to {output_path}")

def train_fold(model, train_loader, tune_loader, optimizer, device, epochs, 
               primary_weight=1.0, aux_weight=1.0, contrast_weight=0.5, trial=None):
    """
    Train the model on the training set and evaluate on the tune set.
    Computes:
      - Primary loss: BCEWithLogitsLoss for binary classification.
      - Auxiliary loss: CrossEntropyLoss for cancer type classification.
      - Contrastive loss on bag-level representations (using auxiliary labels).
    Total loss is the weighted sum of these losses.
    Uses mixed precision training via torch.cuda.amp.
    Intermediate validation ROC-AUC is reported to the trial (if provided).
    """
    model.train()
    primary_loss_fn = nn.BCEWithLogitsLoss()
    aux_loss_fn = nn.CrossEntropyLoss()
    best_val_auc = 0.0
    scaler = GradScaler()  # For mixed precision
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_contrast_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            batch_aux_labels = []
            bag_reps = []
            batch_primary_labels_int = [] 
            batch_loss = 0.0
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):  # Mixed precision forward pass
                for sample in batch:
                    if len(sample) == 3:
                        embedding, primary_label, aux_label = sample
                    else:
                        embedding, primary_label = sample
                        aux_label = 0
                    embedding = embedding.to(device).float()
                    bag_rep, binary_logit, aux_logits, _ = model(embedding)
                    batch_primary_labels_int.append(torch.tensor([int(primary_label)], device=device))

                    batch_aux_labels.append(torch.tensor([int(aux_label)], device=device))
                    bag_reps.append(bag_rep.unsqueeze(0))
                    primary_loss = primary_loss_fn(binary_logit, torch.tensor(float(primary_label), device=device))
                    try:
                        aux_val = int(aux_label)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid auxiliary label '{aux_label}' encountered; setting default value 0.")
                        aux_val = 0
                    aux_label_tensor = torch.tensor([int(aux_label)], device=device)  # shape [1]
                    aux_loss = aux_loss_fn(aux_logits, aux_label_tensor)
                    sample_loss = primary_weight * primary_loss + aux_weight * aux_loss
                    batch_loss += sample_loss
                if len(bag_reps) > 1:
                    bag_reps_tensor = torch.cat(bag_reps, dim=0)  # [batch_size, M]
                    aux_labels_tensor = torch.cat(batch_aux_labels, dim=0)
                    primary_labels_tensor = torch.cat(batch_primary_labels_int, dim=0)
                    contrast_loss = supervised_contrastive_loss(bag_reps_tensor, primary_labels_tensor)
                else:
                    contrast_loss = torch.tensor(0.0, device=device)
                total_loss = batch_loss + contrast_weight * contrast_loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += total_loss.item()
            running_contrast_loss += contrast_loss.item()
            num_batches += 1
        avg_loss = running_loss / num_batches if num_batches else 0.0
        # after computing avg_loss, etc.
        val_loss, val_auc = evaluate_with_loss(
            model, tune_loader, device,
            primary_loss_fn, aux_loss_fn,
            primary_weight=primary_weight,
            aux_weight=aux_weight)

        if dist.get_rank() == 0:
            wandb.log({
                "epoch":           epoch + 1,
                "train_loss":      avg_loss,
                "tune_loss":       val_loss,                   # ← new
                "tune_auc":        val_auc,
                "contrast_loss":   running_contrast_loss/num_batches})

        if trial is not None:
            trial.report(val_auc, epoch)
            if trial.should_prune():
                logger.info(f"Trial pruned at epoch {epoch+1}")
                raise optuna.exceptions.TrialPruned()
        if val_auc > best_val_auc:
            best_val_auc = val_auc
    return best_val_auc

def run_cross_validation(hyperparameters, fold_dir, embeddings_dir, label_col, 
                         case_id_col, aux_label_col, device, epochs, batch_size):
    """
    Runs 3-fold cross validation.
    First, computes a global auxiliary mapping dictionary across all folds.
    For each fold, builds the lazy dataset info from CSV files, constructs DataLoaders, 
    trains the multi-task model, and plots the ROC curve on the test set.
    Returns the average test ROC-AUC over all folds.
    """
    global best_auc_global, best_model_path                            # ★
    best_model_this_trial = None                                       # ★
    best_auc_this_trial  = 0.0  
    
    aux_mapping = compute_aux_mapping(fold_dir, aux_label_col)
    
    fold_aucs = []
    selected_folds = [int(f) for f in args.folds.split(",")]
    for fold in selected_folds:
        logger.info(f"========== Starting fold-{fold} ==========")
        train_labels_file = os.path.join(fold_dir, f"fold-{fold}", "train_contrastive_updated.csv")
        tune_labels_file = os.path.join(fold_dir, f"fold-{fold}", "tune_contrastive_updated.csv")
        test_labels_file = os.path.join(fold_dir, f"fold-{fold}", "test_contrastive_updated.csv")
        
        train_info = build_dataset_info(train_labels_file, embeddings_dir, label_col, case_id_col, aux_label_col, aux_mapping)
        tune_info = build_dataset_info(tune_labels_file, embeddings_dir, label_col, case_id_col, aux_label_col, aux_mapping)
        test_info = build_dataset_info(test_labels_file, embeddings_dir, label_col, case_id_col, aux_label_col, aux_mapping)
        
        train_dataset = MemmapDataset(
            train_info,
            dat_path  = "/data/temporary/amirhosein/UNI_processed/all_embs.dat",
            index_path= "/data/temporary/amirhosein/UNI_processed/offsets.pkl"
            )
        tune_dataset = LazyEmbeddingsDataset(tune_info)
        test_dataset = LazyEmbeddingsDataset(test_info)
        
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        tune_loader = data.DataLoader(tune_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        train_sampler = DistributedSampler(train_dataset)
        train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   sampler=train_sampler,
                                   num_workers=16,
                                   pin_memory=True,
                                   prefetch_factor=4,
                                   persistent_workers=True,
                                   collate_fn=collate_fn)
        model = MultiTaskMILClassifier(
            input_dim=1024,
            hidden_dim=hyperparameters["hidden_dim"],
            M=hyperparameters["M"],
            L=hyperparameters["L"],
            attention_type=hyperparameters["model_choice"],
            aux_num_classes=len(aux_mapping)
        )
        model.to(device)
        model = DDP(model, device_ids=[args.local_rank])
        optimizer = optim.Adam(model.parameters(), lr=hyperparameters["lr"], weight_decay=hyperparameters["reg"])
        
        best_val_auc = train_fold(model, train_loader, tune_loader, optimizer, device, epochs,
                                  primary_weight=1.0, aux_weight=hyperparameters["aux_weight"], 
                                  contrast_weight=hyperparameters["contrast_weight"])
        logger.info(f"Fold-{fold} best tune ROC-AUC: {best_val_auc:.4f}")
        
        test_auc = evaluate_with_loss(model, test_loader, device)
        logger.info(f"Fold-{fold} test ROC-AUC: {test_auc:.4f}")
        fold_aucs.append(test_auc)
        # ------------- keep the best fold-model for this trial ------------
        if test_auc > best_auc_this_trial:                              # ★
            best_auc_this_trial  = test_auc                             # ★
            best_model_this_trial = model                                # ★
        # ------------------------------------------------------------------
        
        all_labels = []
        all_probs = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                for sample in batch:
                    if len(sample) == 3:
                        embedding, primary_label, _ = sample
                    else:
                        embedding, primary_label = sample
                    embedding = embedding.to(device).float()
                    _, binary_logit, _, _ = model(embedding)
                    prob = torch.sigmoid(binary_logit).item()
                    all_probs.append(prob)
                    all_labels.append(float(primary_label))
        roc_output_path = f"roc_curve_fold_{fold}.png"
        plot_roc(all_labels, all_probs, roc_output_path)
        wandb.log({f"roc_curve_fold_{fold}": wandb.Image(roc_output_path)})
    average_auc = np.mean(fold_aucs)
    logger.info(f"Average test ROC-AUC over folds: {average_auc:.4f}")

    # ------------- save model if we beat the global record ---------------
    if average_auc > best_auc_global:                                   # ★
        best_auc_global = average_auc                                   # ★
        if dist.get_rank() == 0:
            torch.save(best_model_this_trial.state_dict(), best_model_path) # ★
            print(f"New best model saved (AUC = {average_auc:.4f})")         # ★
    # ---------------------------------------------------------------------

    return average_auc

# -------------------------
# Objective Function for Hyperparameter Optimization
# -------------------------
def objective(trial):
    logger.info("Entering the `objective` function")
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    reg = trial.suggest_loguniform("reg", 1e-7, 1e-4)
    hidden_dim = trial.suggest_categorical("hidden_dim", [1024])
    M = trial.suggest_categorical("M", [1024])
    L = trial.suggest_categorical("L", [64, 256])
    model_choice = trial.suggest_categorical("model_choice", ["attention"])
    epochs = trial.suggest_int("epochs", 6 , 10)
    aux_weight = trial.suggest_float("aux_weight", 0.0 , 0.0)
    contrast_weight = trial.suggest_float("contrast_weight", 0.0, 0.0)
    batch_size = trial.suggest_categorical("batch_size", [8])
    
    hyperparameters = {
        "lr": lr,
        "reg": reg,
        "hidden_dim": hidden_dim,
        "M": M,
        "L": L,
        "model_choice": model_choice,
        "aux_weight": aux_weight,
        "contrast_weight": contrast_weight,
    }
    # ------------------------------------------------------------------
    # Add a clear logging block here:
    logger.info("==============================================")
    logger.info("Starting a new trial with hyperparameters:")
    for key, val in hyperparameters.items():
        logger.info(f"  {key}: {val}")
    logger.info(f"  epochs: {epochs}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info("==============================================")
    # ------------------------------------------------------------------
    if dist.get_rank() == 0:
        wandb.init(project="baseline line 6 cancer DDP", config=hyperparameters, reinit=True)
    else:
        wandb.init(mode="disabled")
    global args
    device = torch.device("cuda")
    fold_dir = args.fold_dir
    embeddings_dir = args.embeddings_dir
    label_col = args.label_col
    case_id_col = args.case_id_col
    aux_label_col = args.aux_label_col

    average_auc = run_cross_validation(
        hyperparameters, fold_dir, embeddings_dir, label_col, case_id_col, aux_label_col,
        device, epochs, batch_size
    )
    if dist.get_rank() == 0:
        wandb.log({"average_test_auc": average_auc})
    wandb.finish()
    return 1 - average_auc  # Since we maximize ROC-AUC, we minimize (1 - AUC)

# -------------------------
# Main Function and Argument Parsing
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Task MIL with Contrastive Loss, Bayesian Optimization, and 3-fold CV (Mixed Precision Training)"
    )
    parser.add_argument("--fold_dir", type=str, required=False,
                         default="/data/temporary/amirhosein/UNI_processed/labels/all_cancer/",
                        help="Directory containing fold-0, fold-1, and fold-2 subdirectories with label CSV files.")
    parser.add_argument("--embeddings_dir", type=str, required=False,
                        default="/data/temporary/amirhosein/UNI_processed/model_script_concatenated/",
                        help="Directory containing pre-computed .pt embedding files.")
    parser.add_argument("--label_col", type=str, default="TP53",
                        help="Column name for primary label in CSV files (default: label).")
    parser.add_argument("--case_id_col", type=str, default="case_id",
                        help="Column name for case identifier in CSV files (default: case_id).")
    parser.add_argument("--aux_label_col", type=str, default="cancer",
                        help="Column name for auxiliary label (cancer type) in CSV files (default: cancer_type).")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="Enable CUDA if available.")
    parser.add_argument("--n_trials", type=int, default=8,
                        help="Number of hyperparameter optimization trials (default: 20).")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="DDP: GPU index for this process")
    parser.add_argument(
    "--folds", type=str, default="0",
    help="Comma-separated list of folds to run (e.g. '1' or '0,2')")
    if 'ipykernel' in sys.argv[0]:
        sys.argv.extend([
            '--fold_dir', "/data/temporary/amirhosein/UNI_processed/labels/all_cancer/",
            '--embeddings_dir', "/data/temporary/amirhosein/UNI_processed/model_script_concatenated/"
        ])
    
    
    
    args, _ = parser.parse_known_args()
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")

    if dist.get_rank() == 0:
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(),
                                    pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=args.n_trials)

        logger.info("========== Best Trial ==========")
        best_trial = study.best_trial
        logger.info(f"Value (1 - avg ROC-AUC): {best_trial.value:.4f}")
        logger.info("Best hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")

        print("Best trial:")
        print("  Value: ", best_trial.value)
        print("  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")