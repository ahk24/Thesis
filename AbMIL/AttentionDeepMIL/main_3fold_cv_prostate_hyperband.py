#!/usr/bin/env python3
from __future__ import print_function
import logging
import argparse
import numpy as np
import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, roc_auc_score

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
            prob = Y_prob.mean().item()

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

    data_root = "/data/temporary/projects/mutation-prediction/csvs/3-folds-prostate-only"
    embedding_dirs = "/data/temporary/amirhosein/model_script_concatenated"

    # Default hyperparameters (will be overridden by Hyperband if used)
    model_type = "attention"  # or "gated_attention"
    input_dim = 1024
    hidden_dim = 500
    attn_dim = 128
    attn_branches = 1
    epochs = 10
    lr = 0.0005
    weight_decay = 1e-5
    patience = 7  # early stopping patience

    fold_metrics = {'precision': [], 'recall': [], 'auc': []}

    # 3-fold cross-validation loop
    for fold_idx in range(0, 3):
        logging.info("=== Starting FOLD %d ===", fold_idx)
        fold_path = os.path.join(data_root, f"fold-{fold_idx}")

        train_csv = os.path.join(fold_path, "train.csv")
        val_csv   = os.path.join(fold_path, "tune.csv")
        test_csv  = os.path.join(fold_path, "test.csv")

        logging.info("Loading datasets for fold %d", fold_idx)
        train_dataset = PreConcatenatedDataset(embedding_dirs, train_csv)
        val_dataset   = PreConcatenatedDataset(embedding_dirs, val_csv)
        test_dataset  = PreConcatenatedDataset(embedding_dirs, test_csv)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

        logging.info("Initializing model for fold %d", fold_idx)
        if model_type == "attention":
            model = Attention(input_dim=input_dim, M=hidden_dim, L=attn_dim, attention_branches=attn_branches)
        else:
            model = GatedAttention(input_dim=input_dim, M=hidden_dim, L=attn_dim, attention_branches=attn_branches)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        best_state_dict = None
        no_improvement_counter = 0

        for epoch_idx in range(1, epochs + 1):
            logging.info("[Fold %d | Epoch %d] Starting training", fold_idx, epoch_idx)
            train_loss, _ = train_one_epoch(model, train_loader, optimizer, device)
            logging.info("[Fold %d | Epoch %d] Training complete. Loss: %.4f", fold_idx, epoch_idx, train_loss)
            
            logging.info("[Fold %d | Epoch %d] Starting validation", fold_idx, epoch_idx)
            val_loss, _, _, _ = evaluate(model, val_loader, device)
            logging.info("[Fold %d | Epoch %d] Validation complete. Loss: %.4f", fold_idx, epoch_idx, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = model.state_dict()
                no_improvement_counter = 0
                logging.info("[Fold %d | Epoch %d] New best model found (Validation Loss: %.4f)", fold_idx, epoch_idx, best_val_loss)
            else:
                no_improvement_counter += 1
                logging.info("[Fold %d | Epoch %d] No improvement. Counter: %d", fold_idx, epoch_idx, no_improvement_counter)

            if no_improvement_counter >= patience:
                logging.info("[Fold %d] Early stopping triggered after %d epochs with no improvement.", fold_idx, patience)
                break

        if best_state_dict:
            model.load_state_dict(best_state_dict)
            logging.info("Best model loaded for fold %d", fold_idx)

        logging.info("Starting testing for fold %d", fold_idx)
        test_loss, _, y_true, y_scores = evaluate(model, test_loader, device)
        logging.info("[Fold %d] Testing complete. Loss: %.4f", fold_idx, test_loss)

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

    avg_precision = sum(fold_metrics['precision']) / 3
    avg_recall    = sum(fold_metrics['recall'])    / 3
    valid_aucs    = [a for a in fold_metrics['auc'] if not (isinstance(a, float) and (a != a))]
    avg_auc       = sum(valid_aucs) / len(valid_aucs) if valid_aucs else float('nan')

    logging.info("=== 3-FOLD CROSS-VALIDATION RESULTS ===")
    logging.info("Average Precision: %.4f", avg_precision)
    logging.info("Average Recall:    %.4f", avg_recall)
    logging.info("Average AUC:       %.4f", avg_auc)
    logging.info("Script finished successfully")

##############################################
#        Hyperband Integration Below         #
##############################################

def sample_config():
    config = {}
    config["lr"] = 10 ** np.random.uniform(-5, -3)
    config["weight_decay"] = 10 ** np.random.uniform(-6, -3)
    config["hidden_dim"] = int(np.random.uniform(250, 750))
    config["attn_dim"] = int(np.random.uniform(64, 256))
    config["model_type"] = np.random.choice(["attention", "gated_attention"])
    return config

def run_cv_experiment(config, max_epochs):
    logging.info("Running CV experiment with config: %s and max_epochs: %d", config, max_epochs)
    input_dim = 1024
    attn_branches = 1
    patience = 7
    data_root = "/data/temporary/projects/mutation-prediction/csvs/3-folds-prostate-only"
    embedding_dirs = "/data/temporary/amirhosein/model_script_concatenated"
    
    fold_val_losses = []
    
    for fold_idx in range(0, 3):
        logging.info("=== [Hyperband] Starting FOLD %d ===", fold_idx)
        fold_path = os.path.join(data_root, f"fold-{fold_idx}")
        train_csv = os.path.join(fold_path, "train.csv")
        val_csv   = os.path.join(fold_path, "tune.csv")
        test_csv  = os.path.join(fold_path, "test.csv")
        
        logging.info("Loading datasets for fold %d", fold_idx)
        train_dataset = PreConcatenatedDataset(embedding_dirs, train_csv)
        val_dataset   = PreConcatenatedDataset(embedding_dirs, val_csv)
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
        
        model_type = config.get("model_type", "attention")
        hidden_dim = config.get("hidden_dim", 500)
        attn_dim = config.get("attn_dim", 128)
        lr = config.get("lr", 0.0005)
        weight_decay = config.get("weight_decay", 1e-5)
        
        logging.info("Initializing model for fold %d with config: %s", fold_idx, config)
        if model_type == "attention":
            model = Attention(input_dim=input_dim, M=hidden_dim, L=attn_dim, attention_branches=1)
        else:
            model = GatedAttention(input_dim=input_dim, M=hidden_dim, L=attn_dim, attention_branches=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_loss = float("inf")
        no_improvement_counter = 0
        
        for epoch_idx in range(1, max_epochs + 1):
            logging.info("[Hyperband | Fold %d | Epoch %d] Starting training", fold_idx, epoch_idx)
            train_loss, _ = train_one_epoch(model, train_loader, optimizer, device)
            logging.info("[Hyperband | Fold %d | Epoch %d] Training complete. Loss: %.4f", fold_idx, epoch_idx, train_loss)
            
            logging.info("[Hyperband | Fold %d | Epoch %d] Starting validation", fold_idx, epoch_idx)
            val_loss, _, _, _ = evaluate(model, val_loader, device)
            logging.info("[Hyperband | Fold %d | Epoch %d] Validation complete. Loss: %.4f", fold_idx, epoch_idx, val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
            if no_improvement_counter >= patience:
                logging.info("[Hyperband | Fold %d] Early stopping triggered.", fold_idx)
                break
        fold_val_losses.append(best_val_loss)
    
    avg_val_loss = sum(fold_val_losses) / len(fold_val_losses) if fold_val_losses else float("inf")
    logging.info("CV experiment complete. Average best validation loss: %.4f", avg_val_loss)
    return avg_val_loss

def final_retrain(config):
    logging.info("Starting final retraining with best configuration: %s", config)
    use_cuda = False
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    data_root = "/data/temporary/projects/mutation-prediction/csvs/3-folds-prostate-only"
    embedding_dirs = "/data/temporary/amirhosein/model_script_concatenated"
    input_dim = 1024
    attn_branches = 1
    epochs = 10
    patience = 7
    fold_metrics = {'precision': [], 'recall': [], 'auc': []}
    saved_models = {}
    
    for fold_idx in range(0, 3):
        logging.info("=== Final Retraining: Starting FOLD %d ===", fold_idx)
        fold_path = os.path.join(data_root, f"fold-{fold_idx}")
        train_csv = os.path.join(fold_path, "train.csv")
        val_csv   = os.path.join(fold_path, "tune.csv")
        test_csv  = os.path.join(fold_path, "test.csv")
        
        train_dataset = PreConcatenatedDataset(embedding_dirs, train_csv)
        val_dataset   = PreConcatenatedDataset(embedding_dirs, val_csv)
        test_dataset  = PreConcatenatedDataset(embedding_dirs, test_csv)
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
        
        model_type = config.get("model_type", "attention")
        hidden_dim = config.get("hidden_dim", 500)
        attn_dim = config.get("attn_dim", 128)
        lr = config.get("lr", 0.0005)
        weight_decay = config.get("weight_decay", 1e-5)
        
        if model_type == "attention":
            model = Attention(input_dim=input_dim, M=hidden_dim, L=attn_dim, attention_branches=attn_branches)
        else:
            model = GatedAttention(input_dim=input_dim, M=hidden_dim, L=attn_dim, attention_branches=attn_branches)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_loss = float("inf")
        best_state_dict = None
        no_improvement_counter = 0
        
        for epoch_idx in range(1, epochs + 1):
            logging.info("[Final Retrain Fold %d | Epoch %d] Starting training", fold_idx, epoch_idx)
            train_loss, _ = train_one_epoch(model, train_loader, optimizer, device)
            logging.info("[Final Retrain Fold %d | Epoch %d] Training complete. Loss: %.4f", fold_idx, epoch_idx, train_loss)
            
            logging.info("[Final Retrain Fold %d | Epoch %d] Starting validation", fold_idx, epoch_idx)
            val_loss, _, _, _ = evaluate(model, val_loader, device)
            logging.info("[Final Retrain Fold %d | Epoch %d] Validation complete. Loss: %.4f", fold_idx, epoch_idx, val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = model.state_dict()
                no_improvement_counter = 0
                logging.info("[Final Retrain Fold %d | Epoch %d] New best model (Validation Loss: %.4f)", fold_idx, epoch_idx, best_val_loss)
            else:
                no_improvement_counter += 1
                logging.info("[Final Retrain Fold %d | Epoch %d] No improvement. Counter: %d", fold_idx, epoch_idx, no_improvement_counter)
            if no_improvement_counter >= patience:
                logging.info("[Final Retrain Fold %d] Early stopping triggered.", fold_idx)
                break
        
        if best_state_dict:
            model.load_state_dict(best_state_dict)
            saved_models[fold_idx] = model.state_dict()
            logging.info("Final best model loaded for fold %d", fold_idx)
        logging.info("Starting testing for fold %d", fold_idx)
        test_loss, _, y_true, y_scores = evaluate(model, test_loader, device)
        y_pred = [1 if p >= 0.5 else 0 for p in y_scores]
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        try:
            auc_val = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc_val = float('nan')
        logging.info("[Final Retrain Fold %d] Metrics - Precision: %.4f, Recall: %.4f, AUC: %.4f", fold_idx, precision, recall, auc_val)
        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['auc'].append(auc_val)
        
    avg_precision = sum(fold_metrics['precision']) / 3
    avg_recall = sum(fold_metrics['recall']) / 3
    valid_aucs = [a for a in fold_metrics['auc'] if not (isinstance(a, float) and (a != a))]
    avg_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else float('nan')
    
    logging.info("=== Final Retraining 3-FOLD CV RESULTS ===")
    logging.info("Average Precision: %.4f", avg_precision)
    logging.info("Average Recall:    %.4f", avg_recall)
    logging.info("Average AUC:       %.4f", avg_auc)
    
    for fold_idx, state_dict in saved_models.items():
        model_file = f"best_model_fold_{fold_idx}.pt"
        torch.save(state_dict, model_file)
        logging.info("Saved best model for fold %d to %s", fold_idx, model_file)
    return saved_models, fold_metrics

def hyperband_tuning():
    R = 10
    eta = 3
    s_max = int(np.floor(np.log(R) / np.log(eta)))
    best_config = None
    best_score = float("inf")
    all_results = []

    for s in reversed(range(s_max + 1)):
        n = int(np.ceil((s_max + 1) / (s + 1)) * (eta ** s))
        r = R * (eta ** (-s))
        logging.info("Hyperband bracket s=%d: %d configurations, starting with r=%d epochs", s, n, int(r))
        T = [sample_config() for _ in range(n)]
        for i in range(s + 1):
            n_i = int(np.floor(n * (eta ** (-i))))
            r_i = int(r * (eta ** i))
            logging.info("Hyperband bracket s=%d, round %d: evaluating %d configurations for %d epochs", s, i, n_i, r_i)
            scores = []
            new_T = []
            for config in T:
                score = run_cv_experiment(config, max_epochs=r_i)
                scores.append(score)
                new_T.append(config)
                logging.info("Config %s achieved score %.4f", config, score)
            indices = np.argsort(scores)
            num_keep = max(int(np.floor(n_i / eta)), 1)
            T = [new_T[i] for i in indices[:num_keep]]
            if i == s:
                best_bracket_score = scores[indices[0]]
                best_bracket_config = new_T[indices[0]]
                all_results.append((best_bracket_config, best_bracket_score))
                if best_bracket_score < best_score:
                    best_score = best_bracket_score
                    best_config = best_bracket_config
        logging.info("Finished bracket s=%d", s)
    logging.info("Hyperband tuning complete. Best config: %s with score %.4f", best_config, best_score)
    
    saved_models, final_metrics = final_retrain(best_config)
    logging.info("Final retraining complete with best configuration.")
    return best_config, saved_models, final_metrics

##############################################
#            End of Hyperband                #
##############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_hyperband", action="store_true", help="Use Hyperband hyperparameter tuning")
    args, unknown = parser.parse_known_args()

    if args.use_hyperband:
        hyperband_tuning()
    else:
        main()
