# This will make new datamodules that inherit from AbsDatamodule for classification purposes
import functools
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from pysmilesutils.augment import SMILESAugmenter
from pysmilesutils.datautils import TokenSampler
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from pysmilesutils.datautils import ChunkBatchSampler

from Chemformer.molbart.data.util import BatchEncoder, build_attention_mask, build_target_mask
from Chemformer.molbart.utils.tokenizers import ChemformerTokenizer, TokensMasker
from Chemformer.molbart.data.base import _AbsDataModule

class B3DBDataModule(_AbsDataModule):
    """
    DataModule for BBB permeability classification using the B3DB dataset.
    
    Expects a CSV file with columns:
      - "smiles": SMILES string for each molecule
      - "label": binary label (either "BBB+" or "BBB-")
    
    The labels are converted to binary integers (e.g. 1 for "BBB+", 0 for "BBB-").
    The module tokenizes the SMILES (using a provided ChemformerTokenizer) and returns a batch
    that contains only encoder inputs (and its corresponding pad mask) plus the labels.
    """
    
    def __init__(self, **kwargs):
        """
        Additional keyword arguments should include at least:
          - dataset_path: path to the CSV file
          - tokenizer: an instance of ChemformerTokenizer
          - batch_size: batch size to use
          - max_seq_len: maximum sequence length for tokenization
          - unified_model: (optional) if True, add a separator token at the end of SMILES.
          
        Other parameters such as val_idxs, test_idxs, split_perc etc. can also be passed.
        """
        super().__init__(**kwargs)
        # For classification we only need to encode the SMILES (no masking on the encoder needed)
        self._encoder = BatchEncoder(tokenizer=self.tokenizer, masker=None, max_seq_len=self.max_seq_len)
    
    def _load_all_data(self) -> None:
        """
        Load the B3DB CSV file. The CSV is expected to have columns "smiles" and "label".
        Here, the "label" column is converted to integers (1 for "BBB+", 0 for "BBB-").
        """
        df = pd.read_csv(self.dataset_path, sep="\t")
        # Convert the string labels to binary (customize as needed)
        labels = [1 if lab.strip() == "BBB+" else 0 for lab in df["BBB+/BBB-"].tolist()]
        self._all_data = {
            "smiles": df["SMILES"].tolist(),
            "label": labels,
        }
    
        print(f"Loaded {len(self._all_data['smiles'])} SMILES strings.")
        print(f"Number of BBB⁺ samples: {sum(self._all_data['label'])}")
        print(f"Number of BBB⁻ samples: {len(self._all_data['label']) - sum(self._all_data['label'])}")
        print(f"Number of labels: {len(self._all_data['label'])}")


    def _transform_batch(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform a batch of data (a list of dictionaries) into tokenized inputs and a tensor of labels.
        
        Returns:
            encoder_ids: Token ids for the SMILES strings (shape: [seq_len, batch_size])
            encoder_mask: Attention mask for the SMILES strings (shape: [seq_len, batch_size])
            labels_tensor: LongTensor of shape (batch_size,) with binary labels.
        """
        # Extract SMILES and labels from each batch element
        #task_token = "<CLS>"
        #smiles = [f"{task_token} {item['smiles']}" for item in batch]
        smiles = [item["smiles"] for item in batch]
        labels = [item["label"] for item in batch]
        
        # Tokenize the SMILES using the BatchEncoder. 
        encoder_ids, encoder_mask = self._encoder(
            smiles, mask=False, add_sep_token=self.unified_model
        )
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return encoder_ids, encoder_mask, labels_tensor, smiles
    
    def _collate(self, batch: List[Dict[str, Any]], train: bool = True) -> Dict[str, Any]:
        """
        Collate function used by the DataLoader. For classification, we only need the encoder inputs and labels.
        """
        encoder_ids, encoder_mask, labels, smiles = self._transform_batch(batch, train)
        return {
            "encoder_input": encoder_ids,
            "encoder_pad_mask": encoder_mask,
            "labels": labels,
            "smiles": [item['smiles'] for item in batch],
        }
    
class BBBPDataModule(_AbsDataModule):
    """
    DataModule for loading the MoleculeNet BBBP dataset.
    
    The MoleculeNet BBBP CSV is expected to contain:
      - "smiles": SMILES string for each molecule.
      - "p_np": Binary label indicating permeability (1 for BBB⁺, 0 for BBB⁻).
      
    The module tokenizes the SMILES using the provided ChemformerTokenizer and returns batches
    that contain only encoder inputs (and pad masks) along with the binary labels.
    """
    def __init__(self, **kwargs):
        """
        Expects at minimum:
          - dataset_path: Path to the MoleculeNet BBBP CSV file.
          - tokenizer: An instance of ChemformerTokenizer.
          - batch_size: Batch size.
          - max_seq_len: Maximum sequence length for tokenization.
          - unified_model: (Optional) If True, a separator token is added.
          
        Additional parameters such as val_idxs, test_idxs, split_perc, etc. can also be passed.
        """
        super().__init__(**kwargs)
        # For classification, we only need to encode the SMILES.
        self._encoder = BatchEncoder(tokenizer=self.tokenizer, masker=None, max_seq_len=self.max_seq_len)
    
    def _load_all_data(self) -> None:
        """
        Load the MoleculeNet BBBP CSV file.
        Expected columns:
          - "smiles": SMILES strings.
          - "p_np": Binary labels (1 for BBB⁺, 0 for BBB⁻).
          
        If the CSV contains additional columns, they will be ignored.
        """
        df = pd.read_csv(self.dataset_path)
        # Ensure the required columns exist.
        if "smiles" not in df.columns or "p_np" not in df.columns:
            raise ValueError("CSV must contain columns 'smiles' and 'p_np'.")
        # Read the SMILES strings and labels.
        smiles = df["smiles"].tolist()
        labels = df["p_np"].tolist()  # assumes labels are already 0 or 1
        self._all_data = {
            "smiles": smiles,
            "label": labels,
        }
    
    def _transform_batch(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform a batch into tokenized encoder inputs and a tensor of labels.
        
        Returns:
            encoder_ids: Token IDs for the SMILES strings (shape: [seq_len, batch_size]).
            encoder_mask: Attention mask for the SMILES strings (shape: [seq_len, batch_size]).
            labels_tensor: LongTensor of shape (batch_size,) with binary labels.
        """
        # Extract SMILES and labels.
        #task_token = "<CLS>"
        #smiles = [f"{task_token} {item['smiles']}" for item in batch]
        smiles = [item["smiles"] for item in batch]
        labels = [item["label"] for item in batch]
        # Tokenize the SMILES. Here, no masking is applied.
        encoder_ids, encoder_mask = self._encoder(smiles, mask=False, add_sep_token=self.unified_model)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return encoder_ids, encoder_mask, labels_tensor, smiles #I added smiles here (if error remove this)
    
    def _collate(self, batch: List[Dict[str, Any]], train: bool = True) -> Dict[str, Any]:
        encoder_ids, encoder_mask, labels, smiles = self._transform_batch(batch, train)
        return {
            "encoder_input": encoder_ids,
            "encoder_pad_mask": encoder_mask,
            "labels": labels,
            "smiles": [item['smiles'] for item in batch], # I added smiles here (if error remove this)
        }


class NeuToxDataModule(_AbsDataModule):
    """
    DataModule for loading the NeuTox Dataset
    
    The NeuTox CSV is expected to contain:
      - "SMILES": SMILES string for each molecule.
      - "Active": Binary label indicating permeability (1 for BBB⁺, 0 for BBB⁻).
      
    The module tokenizes the SMILES using the provided ChemformerTokenizer and returns batches
    that contain only encoder inputs (and pad masks) along with the binary labels.
    """
    def __init__(self, **kwargs):
        """
        Expects at minimum:
          - dataset_path: Path to the MoleculeNet BBBP CSV file.
          - tokenizer: An instance of ChemformerTokenizer.
          - batch_size: Batch size.
          - max_seq_len: Maximum sequence length for tokenization.
          - unified_model: (Optional) If True, a separator token is added.
          
        Additional parameters such as val_idxs, test_idxs, split_perc, etc. can also be passed.
        """
        super().__init__(**kwargs)
        # For classification, we only need to encode the SMILES.
        self._encoder = BatchEncoder(tokenizer=self.tokenizer, masker=None, max_seq_len=self.max_seq_len)
    
    def _load_all_data(self) -> None:
        """
        Load the MoleculeNet BBBP CSV file.
        Expected columns:
          - "smiles": SMILES strings.
          - "active": Binary labels (1 for BBB⁺, 0 for BBB⁻).
          
        If the CSV contains additional columns, they will be ignored.
        """
        df = pd.read_csv(self.dataset_path)
        # Ensure the required columns exist.
        if "SMILES" not in df.columns or "active" not in df.columns:
            raise ValueError("CSV must contain columns 'smiles' and 'p_np'.")
        # Read the SMILES strings and labels.
        smiles = df["SMILES"].tolist()
        labels = df["active"].tolist()  # assumes labels are already 0 or 1
        self._all_data = {
            "smiles": smiles,
            "label": labels,
        }
    
    def _transform_batch(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform a batch into tokenized encoder inputs and a tensor of labels.
        
        Returns:
            encoder_ids: Token IDs for the SMILES strings (shape: [seq_len, batch_size]).
            encoder_mask: Attention mask for the SMILES strings (shape: [seq_len, batch_size]).
            labels_tensor: LongTensor of shape (batch_size,) with binary labels.
        """
        # Extract SMILES and labels.
        task_token = "<CLS>"
        #smiles = [f"{task_token} {item['smiles']}" for item in batch]
        smiles = [item["smiles"] for item in batch]
        labels = [item["label"] for item in batch]
        # Tokenize the SMILES. Here, no masking is applied.
        encoder_ids, encoder_mask = self._encoder(smiles, mask=False, add_sep_token=self.unified_model)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return encoder_ids, encoder_mask, labels_tensor, smiles #I added smiles here (if error remove this)
    
    def _collate(self, batch: List[Dict[str, Any]], train: bool = True) -> Dict[str, Any]:
        encoder_ids, encoder_mask, labels, smiles = self._transform_batch(batch, train)
        return {
            "encoder_input": encoder_ids,
            "encoder_pad_mask": encoder_mask,
            "labels": labels,
            "smiles": [item['smiles'] for item in batch], # I added smiles here (if error remove this)
        }   
