# This inherits from TransformerModels to build a classifier that only takes frrom the encoder
import math
from functools import partial
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from Chemformer.molbart.models import _AbsTransformerModel
from Chemformer.molbart.models.util import PreNormDecoderLayer, PreNormEncoderLayer

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import csv

class ClassifierModel(_AbsTransformerModel):
    def __init__(
        self,
        pad_token_idx,
        vocabulary_size,
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule="cycle",
        warm_up_steps=None,
        dropout=0.1,
        num_classes=2,
        **kwargs,
    ):
        super().__init__(
            pad_token_idx,
            vocabulary_size,
            d_model,
            num_layers,
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            max_seq_len,
            schedule,
            warm_up_steps,
            dropout,
            **kwargs,
        )

        # Remove decoder components for classification task
        self.encoder = nn.TransformerEncoder(
            PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation),
            num_layers,
            norm=nn.LayerNorm(d_model),
        )
 
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # originally the output of the classifier is num_classes, but I have changed it to 1 to see if that helps anything at all
        #self.loss_function = nn.CrossEntropyLoss()  #commented this out bc it wants 2 outputs
        self.loss_function = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with logits for binary classification

        self._init_params()


    def forward(self, x):
        """
        Forward pass for classification.

        Args:
            x (dict): Must contain:
                - "encoder_input": Tensor of token IDs (seq_len, batch_size)
                - "encoder_pad_mask": Bool tensor indicating padded elements (seq_len, batch_size)

        Returns:
            logits (Tensor): (batch_size, num_classes) predictions
        """
        encoder_input = x["encoder_input"]
        #attention_mask = x["attention_mask"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)  # Transformer expects (batch_size, seq_len)
        encoder_embs = self._construct_input(encoder_input)


        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask) 

        # Average Pooling:
        mask = (~encoder_pad_mask).float()          # (batch, seq_len)
        mask = mask.unsqueeze(-1)                   # (batch, seq_len, 1)

        mem_T = memory.transpose(0, 1)              # (batch, seq_len, d_model)
        summed   = (mem_T * mask).sum(dim=1)        # (batch, d_model)
        lengths  = mask.sum(dim=1).clamp(min=1e-6)  
        pooled_output = summed / lengths            # (batch, d_model)


        # Just Using <CLS> token:
        cls_output = memory[0]

        #Combined cls_output and pooled_output
        #comb_output = torch.cat((cls_output, pooled_output), dim=1)

        logits = self.classifier(pooled_output)
        
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        labels = batch["labels"]
        loss = self.loss_function(logits, labels.float().unsqueeze(1))  # Unsqueeze to match logits shape #originally labels.float().unsqueeze(1)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval() 

        with torch.no_grad():
            logits = self.forward(batch)
            labels = batch["labels"]

            loss = self.loss_function(logits, labels.float().unsqueeze(1))  # for BCE use labels.float().unsqueeze(1)

            probs = torch.sigmoid(logits)  
            preds = (probs > 0.5).float()   
            acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy()) # this is the accuracy score for BCE loss

            #This is used with cross-entropy loss
            #preds = torch.argmax(logits, dim=1)
            #probs = torch.softmax(logits, dim=1)
            #acc = (preds == labels).float().mean()

            try:
                roc = roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy()) 
            except Exception as e:
                print(f"ROC_AUC calculation failed: {e}")
                roc = 0.0

            try:
                prc = average_precision_score(labels.cpu().numpy(), probs.cpu().numpy())
            except Exception as e:
                print(f"PRC_AUC calculation failed: {e}")
                prc = 0.0

            self.log("val_BCE_loss", loss, prog_bar=True, on_epoch=True)
            self.log("val_acc", acc, prog_bar=True, on_epoch=True)
            self.log("val_roc_auc", roc, prog_bar=True, on_epoch=True)
            self.log("val_pr_auc", prc, prog_bar=True, on_epoch=True)

        return {"val_loss": loss, "val_acc": acc}


    def test_step(self, batch, batch_idx):
        self.eval()

        with torch.no_grad():
            logits = self.forward(batch)
            labels = batch["labels"]

            loss = self.loss_function(logits, labels.float().unsqueeze(1))  # for BCE use labels.float().unsqueeze(1)
            #preds = torch.argmax(logits, dim=1)


            #acc = (preds == labels).float().mean()
    
            probs = torch.sigmoid(logits)  
            preds = (probs > 0.5).float()  # Convert probabilities to binary predictions 
            acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy()) # this is the accuracy score for BCE loss


            try:
                roc = roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy())
            except Exception as e:
                print(f"ROC_AUC calculation failed: {e}")
                roc = 0.0

            try:
                prc = average_precision_score(labels.cpu().numpy(), probs.cpu().numpy())
            except Exception as e:
                print(f"PRC_AUC calculation failed: {e}")
                prc = 0.0


            self.log("test_loss", loss, prog_bar=True, on_epoch=True)
            self.log("test_acc", acc, prog_bar=True, on_epoch=True)
            self.log("test_roc_auc", roc, prog_bar=True, on_epoch=True)
            self.log("test_pr_auc", prc, prog_bar=True, on_epoch=True)

            return {"test_loss": loss, "test_acc": torch.tensor(acc), "test_roc_auc": torch.tensor(roc), "test_pr_auc": torch.tensor(prc)}

    def test_epoch_end(self, outputs):
        # Aggregate test loss and accuracy over the test set.
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        avg_roc = torch.stack([x["test_roc_auc"] for x in outputs]).mean()
        avg_prc = torch.stack([x["test_pr_auc"] for x in outputs]).mean()
        
        # Log the aggregated metrics
        self.log("avg_test_loss", avg_loss, prog_bar=True)
        self.log("avg_test_acc", avg_acc, prog_bar=True)
        self.log("avg_test_roc_auc", avg_roc, prog_bar=True)
        self.log("avg_test_pr_auc", avg_prc, prog_bar=True)


        metrics = {
            "test_loss": avg_loss.item(),
            "test_acc": avg_acc.item(),
            "test_roc_auc": avg_roc.item(),
            "test_pr_auc": avg_prc.item(),
        }
        

        # Optionally, you can also print them or perform additional computations.
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}")

        return 

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # Linear decay lamba function
        lr_lambda = lambda epoch: 1 - 0.9 * (min(80, epoch) / 80)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    

    def get_attentions(self, batch):
        """Get the attention weights from the model

        Args:
            batch (dict): Input given to model

        Returns:
            attention_weights (Tensor): Attention weights from the model
        """
        enc_input = batch["encoder_input"]
        enc_mask = batch["encoder_pad_mask"]
        enc_attention_mask = batch["attention_mask"]

        encoder_embs = self._construct_input(enc_input)

        memory, attn = self.encoder(encoder_embs, src_key_padding_mask=enc_mask, need_attn=True, attn_mask=enc_attention_mask)

        return memory, attn