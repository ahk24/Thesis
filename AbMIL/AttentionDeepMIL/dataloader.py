#!/usr/bin/env python3
import os
import torch
import pandas as pd
import logging
from torch.utils.data import Dataset

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class PreConcatenatedDataset(Dataset):
    def __init__(self, data_dir, labels_csv, transform=None):
        """
        A Dataset that loads pre-concatenated .pt files.

        Args:
            data_dir (str):
                Directory where pre-concatenated .pt files are saved.
            labels_csv (str):
                CSV with at least:
                  - case_id: to match with the saved .pt files.
                  - TP53: label (0 or 1)
            transform (callable, optional):
                A transform applied to each bag (tensor).
        """
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform

        logging.info("Reading CSV file: %s", labels_csv)
        df = pd.read_csv(labels_csv)
        logging.info("CSV loaded. Total rows: %d", len(df))
        self.samples = []

        for idx, row in df.iterrows():
            if 'case_id' not in row:
                logging.warning("Row %d missing 'case_id', skipping.", idx)
                continue
            case_id = str(row['case_id'])
            if 'TP53' not in row:
                logging.warning("Row %d missing 'TP53'. Skipping.", idx)
                continue
            label_val = float(row['TP53'])
            file_path = os.path.join(data_dir, f"{case_id}.pt")
            if not os.path.isfile(file_path):
                logging.warning("Concatenated file for case_id '%s' not found at '%s'.", case_id, file_path)
                continue

            self.samples.append({
                'case_id': case_id,
                'tp53_label': label_val,
                'file_path': file_path
            })
            logging.debug("Added sample for case_id '%s' from file '%s'.", case_id, file_path)

        logging.info("Dataset initialized with %d valid samples.", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            bag:   (num_instances, embedding_dim)
            label: (1,) with TP53 label
        """
        sample = self.samples[index]
        case_id = sample['case_id']
        tp53_val = sample['tp53_label']
        file_path = sample['file_path']

        label_tensor = torch.tensor([tp53_val], dtype=torch.float32)
        logging.debug("Loading pre-concatenated tensor for case_id '%s' from '%s'", case_id, file_path)
        bag = torch.load(file_path)

        if self.transform is not None:
            logging.debug("Applying transform to bag for case_id '%s'", case_id)
            bag = self.transform(bag)

        logging.debug("Returning bag of shape %s for case_id '%s'", bag.shape, case_id)
        return bag, label_tensor
