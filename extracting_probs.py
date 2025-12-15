import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from Models.MLP import MLPClassifier
from Models.EnsembleClassifier import EnsembleClassifier

results = []          # accumulate rows here

trained_on = ["B3DB", "NeuTox", "BBBP"] #, "B3DB", "NeuTox"
datasets   = ["B3DB", "NeuTox", "BBBP"]
feature_types = ["Descriptors", "Embeddings", "Combined"]
folds = [0, 1, 2, 3, 4]

out_dir = "/fast/grantn2/BBBPermeability/Outputs_Sept2025"
os.makedirs(out_dir, exist_ok=True)

for t_on in trained_on:
    for set_name in datasets:
        for feature_type in feature_types:
            for fold in folds:

                base = f"Output_Files/{t_on}/Avg_Pool/NoLayerNorm"

    
                # load train/val that always come from t_on
                tr_path  = f"{base}/{t_on}/train/{fold}/{feature_type}.csv"
                val_path = f"{base}/{t_on}/val/{fold}/{feature_type}.csv"
                training_set = pd.read_csv(tr_path)
                val_set      = pd.read_csv(val_path)

                # pick the correct test-set        
                if set_name == t_on:
                    test_path = f"{base}/{t_on}/test/{fold}/{feature_type}.csv"
                else:
                    test_path = f"{base}/{set_name}/test/{fold}/{feature_type}.csv"

                test_set = pd.read_csv(test_path)


                scaler = StandardScaler()
                X_train = scaler.fit_transform(training_set.drop(columns=["label"]))
                y_train = training_set["label"].values

                X_val   = scaler.transform(val_set.drop(columns=["label"]))
                y_val   = val_set["label"].values

                X_test  = scaler.transform(test_set.drop(columns=["label"]))
                y_test  = test_set["label"].values

          
                clfs = {
                    "RF":  RandomForestClassifier(n_estimators=100, random_state=42),
                    "LR":  LogisticRegression(max_iter=1000,        random_state=42),
                    "SVM": SVC(kernel="linear", probability=True,   random_state=42),
                    "MLP": MLPClassifier(
                              input_dim=X_train.shape[1],
                              batch_size=16,
                              learning_rate=1e-5,
                              n_epochs=150,
                          )

                }

                # train, predict, store
                for model_name, model in clfs.items():

                    if model_name == "MLP":
                        model.fit(
                            X_train=X_train, y_train=y_train,
                            X_val=X_val,     y_val=y_val
                        )
                        metrics, y_prob = model.evaluate(X_test, y_test)


                    else:
                        model.fit(X_train, y_train)
                        y_prob = model.predict_proba(X_test)[:, 1]  # P(class=1)
                        
                    y_pred = (y_prob >= 0.5).astype(int)

                    # save row-per-sample (long format)
                    fold_df = pd.DataFrame({
                        "trained_on":      t_on,
                        "test_set":        set_name,
                        "feature_type":    feature_type,
                        "fold":            fold,
                        "classifier":      model_name,
                        "label":           y_test,
                        "prob":            y_prob,
                        "pred":            y_pred,
                    })
                    results.append(fold_df)

                print(f"✓ finished {t_on} → {set_name} | {feature_type} | fold {fold}")

        for fold in folds:
            model = EnsembleClassifier(
                        batch_size=16,
                        learning_rate=1e-5,
                        n_epochs=150
                    )
            base = f"Output_Files/{t_on}/Avg_Pool/NoLayerNorm"

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

            scaler_desc = StandardScaler()
            scaler_embed = StandardScaler()
            scaler_concat = StandardScaler()

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

            metrics, y_prob = model.evaluate(
                X_desc=x_test_desc,
                X_embed=x_test_embed,
                X_concat=x_test_concat,
                y_true=y_test,
                device='cpu'
            )

            pd.DataFrame([metrics]).to_csv(f"{out_dir}/Sept9/Ensemble_trained_{t_on}_fold{fold}_metrics_{set_name}.csv", index=False)

            y_pred = (y_prob >= 0.5).astype(int)

            fold_df = pd.DataFrame({
                "trained_on":      t_on,
                "test_set":        set_name,
                "feature_type":    "All",
                "fold":            fold,
                "classifier":      "Ensemble",
                "label":           y_test,
                "prob":            y_prob,
                "pred":            y_pred,
            })
            results.append(fold_df)




results_df = pd.concat(results, ignore_index=True)


results_df.to_csv(f"{out_dir}/all_preds_long_format.csv", index=False)

auroc_table = (
    results_df
    .groupby(["trained_on", "test_set", "feature_type", "classifier", "fold"])
    .apply(lambda g: roc_auc_score(g["label"], g["prob"]))
    .reset_index(name="AUROC")
)
auroc_table.to_csv(f"{out_dir}/foldwise_aurocs.csv", index=False)

print("Done!  Long-format predictions stored; ready for DeLong test.")
