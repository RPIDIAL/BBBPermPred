from cProfile import label
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score, f1_score, matthews_corrcoef, accuracy_score
import Chemformer.molbart.utils.data_utils as util

from Models.ChemformerClassifier import ChemformerClassifier
from Models.MLP import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from rdkit import Chem
from deepchem.feat import RDKitDescriptors


# build three different heads of different dimension using the MLP Classifier as a base

class EnsembleClassifier:
    
    def __init__(self, batch_size=16, learning_rate=1e-4, dropout=0.1, n_epochs=100):
        self.Descriptor_Head = MLPClassifier(input_dim=208, batch_size=batch_size, learning_rate=learning_rate, dropout=dropout, n_epochs=n_epochs)
        self.Embedding_Head = MLPClassifier(input_dim=512, batch_size=batch_size, learning_rate=learning_rate, dropout=dropout, n_epochs=n_epochs)
        self.Concat_Head = MLPClassifier(input_dim=720, batch_size=batch_size, learning_rate=learning_rate, dropout=dropout, n_epochs=n_epochs)

    def fit(self, 
            X_desc, 
            X_embed, 
            X_concat, 
            y, 
            X_val_desc, 
            X_val_embed, 
            X_val_concat, 
            y_val, 
            device='cpu'):
        self.Descriptor_Head.fit(X_desc, y, X_val_desc, y_val, device=device)
        self.Embedding_Head.fit(X_embed, y, X_val_embed, y_val, device=device)
        self.Concat_Head.fit(X_concat, y, X_val_concat, y_val, device=device)


    def predict(self, X_desc, X_embed, X_concat, threshold=0.5, device='cpu'):
        desc_preds, desc_probs = self.Descriptor_Head.predict(X_desc, device=device)
        embed_preds, embed_probs = self.Embedding_Head.predict(X_embed, device=device)
        concat_preds, concat_probs = self.Concat_Head.predict(X_concat, device=device)

        # average the probabilities
        avg_probs = (desc_probs + embed_probs + concat_probs) / 3.0
        preds = (avg_probs >= threshold).astype(int)

        return preds, avg_probs
    

    def evaluate(self, X_desc, X_embed, X_concat, y_true, device='cpu'):
        preds, probs = self.predict(X_desc, X_embed, X_concat, device=device)
        metrics = {
            "accuracy": accuracy_score(y_true, preds),
            "roc_auc": roc_auc_score(y_true, probs),
            "prc_auc": average_precision_score(y_true, probs),
            "f1": f1_score(y_true, preds),
            "mcc": matthews_corrcoef(y_true, preds)
        }

        return metrics, probs 
    

    def get_layer_embeddings(self, X_desc, X_embed, X_concat, y_true, device='cpu'):
        desc_embs, lab1 = self.Descriptor_Head.get_embeddings(X_desc, y_true, device=device)
        embed_embs, lab2 =  self.Embedding_Head.get_embeddings(X_embed, y_true, device=device)
        concat_embs, lab3 = self.Concat_Head.get_embeddings(X_concat, y_true, device=device)

        if (lab1 == lab2).all() and (lab2 == lab3).all():
            label = lab1
        else:
            raise ValueError("Labels from different heads do not match.")


        return desc_embs, embed_embs, concat_embs, lab1

