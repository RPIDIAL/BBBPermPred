# This file uses Chemformer to run classifications
import os
from argparse import Namespace
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader

from Chemformer.molbart.data import DataCollection
import Chemformer.molbart.utils.data_utils as util
from Chemformer.molbart.models import BARTModel, UnifiedModel
from Chemformer.molbart.utils.samplers import BeamSearchSampler
from Chemformer.molbart.utils.tokenizers import ChemformerTokenizer
from Chemformer.molbart.utils import trainer_utils
from Chemformer.molbart.models import Chemformer 

from Models.ClassifierModel import ClassifierModel

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from rdkit import Chem
from deepchem.feat import RDKitDescriptors
from torch import nn


DEFAULT_WEIGHT_DECAY = 0


class ChemformerClassifier(Chemformer):
    """
    Class for building (synthesis) Chemformer model, fine-tuning seq-seq model,
    and predicting/scoring model.
    """
 
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        """
        Args:
            config: OmegaConf config loaded by hydra. Contains the input args of the model,
                including vocabulary, model checkpoint, beam size, etc.

            The config includes the following arguments:
                # Trainer args
                seed: 1
                batch_size: 128
                n_gpus (int): Number of GPUs to use.
                i_chunk: 0              # For inference
                n_chunks: 1             # For inference
                limit_val_batches: 1.0  # For training
                n_buckets: 12           # For training
                n_nodes: 1              # For training
                acc_batches: 1          # For training
                accelerator: null       # For training

                # Data args
                data_path (str): path to data used for training or inference
                backward_predictions (str): path to sampled smiles (for round-trip inference)
                dataset_part (str): Which dataset split to run inference on. ["full", "train", "val", "test"]
                dataset_type (str): The specific dataset type used as input.
                datamodule_type (Optinal[str]): The type of datamodule to build (seq2seq).
                vocabulary_path (str): path to bart_vocabulary.
                task (str): the model task ["forward_prediction", "backward_prediction"]
                data_device (str): device used for handling the data in optimized beam search (use cpu if memor issues).

                # Model args
                model_path (Optional[str]): Path to model weights.
                model_type (str): the model type ["bart", "unified"]
                n_beams (int): Number of beams / predictions from the sampler.
                n_unique_beams (Optional[int]): Restrict number of unique predictions.
                    If None => return all unique solutions.
                train_mode(str): Whether to train the model ("training") or use
                    model for evaluations ("eval").

                train_mode (str): Whether to train the model ("training") or use
                    model for evaluations ("eval").
                device (str): Which device to run model and beam search on ("cuda" / "cpu").
                resume_training (bool): Whether to continue training from the supplied
                    .ckpt file.

                learning_rate (float): the learning rate (for training/fine-tuning)
                weight_decay (float): the weight decay (for training/fine-tuning)

                # Molbart model parameters
                d_model (int): 512
                n_layers (int): 6
                n_heads (int): 8
                d_feedforward (int): 2048

                callbacks: list of Callbacks
                datamodule: the DataModule to use

                # Inference args
                scorers: list of Scores to evaluate sampled smiles against target smiles
                output_score_data: null
                output_sampled_smiles: null
        """

        self.config = config

        self.train_mode = config.train_mode
        print(f"train mode: {self.train_mode}")
        self.train_tokens = config.get("train_tokens")
        self.n_buckets = config.get("n_buckets")
        self.resume_training = False
        if self.train_mode.startswith("train"):
            self.resume_training = config.resume

            if self.resume_training:
                print("Resuming training.")

        device = config.get("device", "cuda")
        data_device = config.get("data_device", "cuda")
        if config.n_gpus < 1:
            device = "cpu"
            data_device = "cpu"

        self.device = device

        self.tokenizer = ChemformerTokenizer(filename=config.vocabulary_path)

        self.model_type = config.model_type
        self.model_path = config.model_path

        self.n_gpus = config.n_gpus
        self.is_data_setup = False
        self.set_datamodule(datamodule_type=config.get("datamodule"))

        print("Vocabulary_size: " + str(len(self.tokenizer)))
        self.vocabulary_size = len(self.tokenizer)

        if self.train_mode.startswith("train"):
            self.train_steps = trainer_utils.calc_train_steps(config, self.datamodule, self.n_gpus)
            print(f"Train steps: {self.train_steps}")

        sample_unique = config.get("n_unique_beams") is not None

        self.sampler = BeamSearchSampler(
            self.tokenizer,
            trainer_utils.instantiate_scorers(self.config.get("scorers")),
            util.DEFAULT_MAX_SEQ_LEN,
            device=device,
            data_device=data_device,
            sample_unique=sample_unique,
        )

        self.build_model(config)
        self.model.num_beams = config.n_beams
        if sample_unique:
            self.model.n_unique_beams = np.min(np.array([self.model.num_beams, config.n_unique_beams]))

        self.trainer = None
        if "trainer" in self.config:
            self.trainer = trainer_utils.build_trainer(config, self.n_gpus)

        self.model = self.model.to(device)
        return
    
    def _random_initialization(
            self, args: Namespace, extra_args: Dict[str, Any], pad_token_idx: int
        ) -> Union[BARTModel, UnifiedModel, ClassifierModel]:
            """
            Constructing a model with randomly initialized weights.

            Args:
                args (Namespace): Grouped model arguments.
                extra_args (Dict[str, Any]): Extra arguments passed to the BARTModel.
                Will be saved as hparams by pytorchlightning.
                pad_token_idx: The index denoting padding in the vocabulary.
            """

            if self.train_mode.startswith("train"):
                total_steps = self.train_steps + 1
            else:
                total_steps = 0

            if self.model_type == "classifier":
            #
                print("Extra args: ", extra_args)
                print("Args: ", args)
                print("Scheduler: ", args.get("schedule"))
                model = ClassifierModel(
                    self.sampler,
                    pad_token_idx,
                    self.vocabulary_size,
                    args.d_model,
                    args.n_layers,
                    args.n_heads,
                    args.d_feedforward,
                    args.get("learning_rate"),
                    DEFAULT_WEIGHT_DECAY,
                    util.DEFAULT_ACTIVATION,
                    total_steps,
                    util.DEFAULT_MAX_SEQ_LEN,
                    schedule=args.get("schedule"),
                    dropout=util.DEFAULT_DROPOUT,
                    warm_up_steps=args.get("warm_up_steps"),
                    **extra_args,
                )

                print("Schedule", model.schedule)

            else:
                raise ValueError(f"Unknown model type []: {self.model_type}")

            return model
    
    def _initialize_from_ckpt(
        self, args: Namespace, extra_args: Dict[str, Any], pad_token_idx: int
    ) -> Union[ClassifierModel, UnifiedModel]:
        """
        Constructing a model with weights from a ckpt-file.

        Args:
            args (Namespace): Grouped model arguments.
            extra_args (Dict[str, Any]): Extra arguments passed to the BARTModel.
            Will be saved as hparams by pytorchlightning.
            pad_token_idx: The index denoting padding in the vocabulary.
        """
        if self.train_mode == "training" or self.train_mode == "train":
            total_steps = self.train_steps + 1

        if self.model_type == "classifier":
            if self.train_mode in ["training", "train"]:
                if self.resume_training:
                    model = ClassifierModel.load_from_checkpoint(
                        self.model_path,
                        strict=False
                    )
                    model.train()
                else:
                    model = ClassifierModel.load_from_checkpoint(
                        self.model_path,
                        pad_token_idx=pad_token_idx,
                        vocabulary_size=self.vocabulary_size,
                        num_steps=total_steps,
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                        schedule=args.schedule,
                        warm_up_steps=args.warm_up_steps,
                        strict=False,   # <-- Pass strict=False here
                        **extra_args,
                    )
            elif self.train_mode in ["validation", "val", "test", "testing", "eval"]:
                model = ClassifierModel.load_from_checkpoint(
                    self.model_path,
                    strict=False
                )
                model.eval()
            else:
                raise ValueError(f"Unknown training mode: {self.train_mode}")
        
        else:
            raise ValueError(f"Unknown model type [bart, unified, classifier]: {self.model_type}")
        return model

    def classify(
            self,
            dataset: str = "full",
            dataloader: Optional[DataLoader] = None,
            return_tokenized: bool = False,
            output_name: str = None,
        ):
        """
        Extract embeddings from the encoder for the specified dataset split,
        classify the samples using the classifier head, and return:
        - embeddings (from the encoder, after pooling),
        - true labels, and
        - classification predictions.
      
        Args:
            dataset (str): Which part of the dataset to use ("train", "val", "test", or "full").
            dataloader (Optional[DataLoader]): If None, the dataloader will be retrieved from self.datamodule.
            return_tokenized (bool): (Not used here, kept for interface consistency.)
        
        Returns:
            embeddings, labels, classification_results
        """
        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        self.model.to(self.device)
        self.model.eval()

        embeddings_list = []
        labels_list = []
        predictions_list = []
        probs_list = []

        for batch in dataloader:
            batch = self.on_device(batch)
            with torch.no_grad():
                encoder_input = batch["encoder_input"]
                encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)

                encoder_embs = self.model._construct_input(encoder_input)
                encoder_output = self.model.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
            
                pooled_emb = encoder_output.mean(dim=0)  # shape:[batch_size, d_model]
            
                # Pass pooled embeddings thru the classifier
                logits = self.model.classifier(pooled_emb)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
        
            embeddings_list.append(pooled_emb.cpu().numpy())
            labels_list.append(batch["labels"].cpu().numpy())
            predictions_list.append(preds.cpu().numpy())
            probs_list.append(probs.cpu().numpy())

        # Concatenate results from all batches.
        embeddings = np.concatenate(embeddings_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        predictions = np.concatenate(predictions_list, axis=0)
        probs = np.concatenate(probs_list, axis=0)

        accuracy = accuracy_score(labels, predictions)
        print(f"Accuracy: {accuracy}")
        roc_auc = roc_auc_score(labels, probs[:, 1])
        print(f"ROC AUC: {roc_auc}")
        prc_auc = average_precision_score(labels, probs[:, 1])
        print(f"PRC AUC: {prc_auc}")
        f1 = f1_score(labels, predictions)
        print(f"F1: {f1}")
        mcc = matthews_corrcoef(labels, predictions)

        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "prc_auc": prc_auc,
            "f1": f1,
            "mcc": mcc
        }

        metrics_df = pd.DataFrame([metrics])
        output_path = self.config.get("output_dir")

        if output_path:
            os.makedirs(output_path, exist_ok=True)
            metrics_df.to_csv(output_path + output_name, index=False)
        else:
            print("No output path provided. Metrics not saved.")

        return embeddings, labels, predictions

    def get_RDKitDescriptors(
        self,
        dataloader: Optional[DataLoader] = None,
        dataset_part: str = "full",
    ):
        """
        Caluclates the RDKit features for a given list of SMILES strings
        Args:
            datmodule: The datamodule to use to load the SMILES strings
        Returns:
            List[np.ndarray]: List of RDKit features
        """

        if dataloader is None:
            dataloader = self.get_dataloader(dataset_part)

        featurizer = RDKitDescriptors()
        features = []

        smiles = []
        for batch in dataloader:
            smiles.extend(batch["smiles"])

        labels = []
        for batch in dataloader:
            labels.extend(batch["labels"].cpu().numpy())

        print(f"Calculating RDKit descriptors for {len(smiles)} SMILES strings.")

        descriptor_length = None
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                descriptor_length = len(featurizer.featurize([mol])[0])
                break
        if descriptor_length is None:
            raise ValueError("No valid molecule found in the dataset.")

        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                feature_vector = featurizer.featurize([mol])[0]
            else:
                feature_vector = np.zeros(descriptor_length)
            features.append(feature_vector)
        return features, labels
     
    def get_combined_features(
        self,
        dataset_part: str = "full",
        layernorm: bool = False,
        preconcat_layernorm: bool = False,
        avg_pooling: bool = True,
        cls: bool = False,
        comb: bool = False,
    ):

        dataloader = self.get_dataloader(dataset_part)
        featurizer = RDKitDescriptors()

        embeddings_list = []
        descriptors_list = []
        labels_list = []
  

        for batch in dataloader:
            batch = self.on_device(batch)

            smiles_batch = batch["smiles"]
            labels_batch = batch["labels"].cpu().numpy()

            descriptor_length = None
            for smi in smiles_batch:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    descriptor_length = len(featurizer.featurize([mol])[0])
                    break
            if descriptor_length is None:
                raise ValueError("No valid molecules were found in the dataset.")

            descriptors_batch = []
            for smi in smiles_batch:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    descriptors = RDKitDescriptors().featurize([mol])[0]
                else:
                    descriptors_batch.append(np.zeros(descriptor_length))
                    continue
                descriptors_batch.append(descriptors)

            descriptors_batch = np.array(descriptors_batch)

            with torch.no_grad():
                encoder_input = batch["encoder_input"]
                encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)

                encoder_embs = self.model._construct_input(encoder_input)
                encoder_output = self.model.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)

                if avg_pooling: 
                    mask = (~encoder_pad_mask).float()          # (batch, seq_len)
                    mask = mask.unsqueeze(-1)                   # (batch, seq_len, 1)

                    mem_T = encoder_output.transpose(0, 1)              # (batch, seq_len, d_model)
                    summed   = (mem_T * mask).sum(dim=1)        # (batch, d_model)
                    lengths  = mask.sum(dim=1).clamp(min=1e-6)  
                    emb = summed / lengths            # (batch, d_model)

                elif cls:
                    emb = encoder_output[0]

                elif comb:
                    mask = (~encoder_pad_mask).float()          # (batch, seq_len)
                    mask = mask.unsqueeze(-1)                   # (batch, seq_len, 1)
                    mem_T = encoder_output.transpose(0, 1)              # (batch, seq_len, d_model)
                    summed   = (mem_T * mask).sum(dim=1)        # (batch, d_model)
                    lengths  = mask.sum(dim=1).clamp(min=1e-6)  
                    pool_emb = summed / lengths            # (batch, d_model)
                    cls_op = encoder_output[0]
                    emb = torch.cat((cls_op, pool_emb), dim=1)


            # Concatenate embeddings and RDKit descriptors
            combined_features = np.concatenate([emb.cpu().numpy(), descriptors_batch], axis=1)
            combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)

            
            embeddings_list.append(emb.cpu().numpy())
            labels_list.append(labels_batch)
            descriptors_list.append(descriptors_batch)

        # Concatenate results from all batches
        x = np.vstack(embeddings_list)
        y = np.concatenate(labels_list) 
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        z = np.vstack(descriptors_list)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        combined_features = np.concatenate([x, z], axis=1)

        if layernorm:
            layernorm_module = nn.LayerNorm(combined_features.shape[1]).to(self.device)
            combined_features_tensor = torch.tensor(combined_features, device=self.device, dtype=torch.float32).to(self.device)
            norm_combined = layernorm_module(combined_features_tensor)
            combined_features = norm_combined.detach().cpu().numpy()

        if preconcat_layernorm:
            desc_layernorm = nn.LayerNorm(z.shape[1]).to(self.device)
            emb_layernorm = nn.LayerNorm(x.shape[1]).to(self.device)
            norm_desc = desc_layernorm(torch.tensor(z, device=self.device, dtype=torch.float32).to(self.device))
            norm_emb = emb_layernorm(torch.tensor(x, device=self.device, dtype=torch.float32).to(self.device))
            x = norm_emb.detach().cpu().numpy()
            z = norm_desc.detach().cpu().numpy()
            combined_features = np.concatenate([x, z], axis=1)

        print("The shapes of all of the the different features are: ")
        print(f"X - the embeddings: {x.shape}")
        print(f"Y - the labels: {y.shape}")
        print(f"Z - the chemical descriptors: {z.shape}")
        print(f"Combined features: {combined_features.shape}")

        return x, y, z, combined_features


        



