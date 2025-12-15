import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from BBBPermPred.Models.MLP import MLPClassifier
from BBBPermPred.Models.EnsembleClassifier import EnsembleClassifier


results = []          # accumulate rows here

trained_on = ["B3DB"] # "NeuTox"
datasets   = ["B3DB", "NeuTox", "BBBP"] # "NeuTox", "BBBP"
feature_types = ["Descriptors"] # "Embeddings", "Combined",
folds = [0, 1, 2, 3, 4]

out_dir = "Outputs_Sept2025/embeddings"
os.makedirs(out_dir, exist_ok=True)

for t_on in trained_on:
    for set_name in datasets:

        base = f"Output_Files/{t_on}/Avg_Pool/NoLayerNorm"

        for fold in folds:
            model = EnsembleClassifier(
                        batch_size=16,
                        learning_rate=1e-5,
                        n_epochs=150
                    )
            base = f"/fast/grantn2/BBBPermeability/Chemformer/Output_Files/{t_on}/Avg_Pool/NoLayerNorm"

            x_train_desc = pd.read_csv(f"{base}/{t_on}/train/{fold}/Descriptors.csv").drop(columns=["label"]).values
            x_train_embed= pd.read_csv(f"{base}/{t_on}/train/{fold}/Embeddings.csv").drop(columns=["label"]).values
            x_train_concat= pd.read_csv(f"{base}/{t_on}/train/{fold}/Combined.csv").drop(columns=["label"]).values

            x_val_desc   = pd.read_csv(f"{base}/{t_on}/val/{fold}/Descriptors.csv").drop(columns=["label"]).values
            x_val_embed  = pd.read_csv(f"{base}/{t_on}/val/{fold}/Embeddings.csv").drop(columns=["label"]).values
            x_val_concat = pd.read_csv(f"{base}/{t_on}/val/{fold}/Combined.csv").drop(columns=["label"]).values

            x_test_desc  = pd.read_csv(f"{base}/{set_name}/test/{fold}/Descriptors.csv").drop(columns=["label"]).values
            x_test_embed = pd.read_csv(f"{base}/{set_name}/test/{fold}/Embeddings.csv").drop(columns=["label"]).values
            x_test_concat= pd.read_csv(f"{base}/{set_name}/test/{fold}/Combined.csv").drop(columns=["label"]).values

            y_train      = pd.read_csv(f"{base}/{t_on}/train/{fold}/Descriptors.csv")["label"].values
            y_val        = pd.read_csv(f"{base}/{t_on}/val/{fold}/Descriptors.csv")["label"].values
            y_test       = pd.read_csv(f"{base}/{set_name}/test/{fold}/Descriptors.csv")["label"].values

            x_full_desc = pd.read_csv(f"{base}/{t_on}/full/{fold}/Descriptors.csv").drop(columns=["label"]).values
            y_full      = pd.read_csv(f"{base}/{t_on}/full/{fold}/Descriptors.csv")["label"].values
            x_full_embed= pd.read_csv(f"{base}/{t_on}/full/{fold}/Embeddings.csv").drop(columns=["label"]).values
            x_full_concat= pd.read_csv(f"{base}/{t_on}/full/{fold}/Combined.csv").drop(columns=["label"]).values

            scaler_desc = MinMaxScaler()
            scaler_embed = MinMaxScaler()
            scaler_concat = MinMaxScaler()

            x_train_desc = scaler_desc.fit_transform(x_train_desc)
            x_train_embed = scaler_embed.fit_transform(x_train_embed)
            x_train_concat = scaler_concat.fit_transform(x_train_concat)

            x_val_desc = scaler_desc.transform(x_val_desc)
            x_val_embed = scaler_embed.transform(x_val_embed)
            x_val_concat = scaler_concat.transform(x_val_concat)

            x_test_desc = scaler_desc.transform(x_test_desc)
            x_test_embed = scaler_embed.transform(x_test_embed)
            x_test_concat = scaler_concat.transform(x_test_concat)

            model.fit(
                X_desc=x_train_desc,
                X_embed=x_train_embed,
                X_concat=x_train_concat,
                y=y_train,
                X_val_desc=x_val_desc,
                X_val_embed=x_val_embed,
                X_val_concat=x_val_concat,
                y_val=y_val,
                device='cpu'
            )

            desc_embs, embed_embs, concat_embs = model.get_layer_embeddings(
                X_desc=x_full_desc,
                X_embed=x_full_embed,
                X_concat=x_full_concat,
                y_true=y_full,
                device='cpu'
            )

            
            out_dir = f"Outputs_Sept2025/embeddings_v2/trained_on_{t_on}/{set_name}/test/{fold}"
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame(y_full).to_csv(f"{out_dir}/labels.csv", index=False)
            pd.DataFrame(desc_embs).to_csv(f"{out_dir}/Descriptors.csv", index=False)
            pd.DataFrame(embed_embs).to_csv(f"{out_dir}/Embeddings.csv", index=False)
            pd.DataFrame(concat_embs).to_csv(f"{out_dir}/Combined.csv", index=False)

